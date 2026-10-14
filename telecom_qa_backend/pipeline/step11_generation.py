from core.schemas import QueryContext, GenerationResponse
from core.interfaces import Generator
from core.tracer import trace_step
from core.structured_llm import generate_structured

def execute(context: QueryContext, generator: Generator) -> QueryContext:
    """
    Step 11: Generation
    If not abstained, prompts the LLM to write the final answer based strictly
    on the validated evidence, and enforces strict citation formatting.
    Injects conversation history for multi-turn context awareness.
    Validates output structure with Pydantic and auto-retries on schema issues.
    """
    if context.is_abstained:
        return context

    query = context.understanding.normalized_query if context.understanding else context.original_query
    input_count = len(context.selected_evidence)

    with trace_step(context, "Step 11: Generation (LLM)", input_count=input_count) as t:
        # Format evidence for the generator, distinguishing 3GPP specs from Web search
        evidence_text = ""
        for chunk in context.selected_evidence:
            meta = chunk.metadata
            is_web = meta.source_type == "web" or meta.ts_number == "Web"
            source_tag = "WEB SEARCH RESULT" if is_web else f"3GPP TS {meta.ts_number} | Rel-{meta.release_version}"
            evidence_text += f"--- EVIDENCE ID: {meta.chunk_id} [{source_tag}] ---\n"
            evidence_text += f"{chunk.text}\n\n"

        # Inject conversation history for multi-turn awareness
        history_text = ""
        if context.conversation_history:
            history_text = "\n\nPREVIOUS CONVERSATION CONTEXT:\n"
            for turn in context.conversation_history[-4:]:  # last 4 turns max
                history_text += f"{turn.role.upper()}: {turn.content}\n"

        prompt = f"""You are an expert telecom 3GPP standards and wireless communication systems engineer.
{history_text}
A user asked: "{query}"

Retrieved evidence (from local 3GPP specifications or web fallback search):
{evidence_text if evidence_text.strip() else "[No direct specification chunks retrieved]"}

Task:
1. If the provided evidence contains relevant technical information, synthesize a comprehensive, rigorous answer. When citing specific evidence chunks, use the Evidence ID in brackets, like: [123e4567-e89b-12d3-a456-426614174000]. Do NOT combine brackets.
2. If the provided evidence is partial, web-based, or does not fully cover all aspects of the user's question, synthesize a detailed, highly technical answer using your broad domain knowledge of telecommunications, wireless communication systems, and 3GPP architectures. If doing this, set "answered_from_general_knowledge" to true.
3. If the query has absolutely NO relation to telecommunications, technology, networking, or science, or you have no technical basis to answer, leave "content" blank.

Return a JSON object matching this schema:
{{
  "content": "The text of your comprehensive technical answer (or blank if completely unanswerable)",
  "citations": ["Evidence-UUID-1", "Evidence-UUID-2"],
  "needs_web_search": false,
  "answered_from_general_knowledge": false
}}

Output strictly valid JSON.
"""

        try:
            gen_res = generate_structured(
                generator=generator,
                prompt=prompt,
                response_model=GenerationResponse,
                max_retries=2,
                context=context.selected_evidence,
            )
            
            if not gen_res.content or gen_res.content.strip() == "":
                context.is_abstained = True
                context.abstention_reason = "Unable to formulate a technically grounded answer."
                context.final_answer = ""
                context.citations = []
            else:
                context.final_answer = gen_res.content
                context.citations = gen_res.citations
                context.needs_web_search = gen_res.needs_web_search
                context.guardrails.answered_from_general_knowledge = gen_res.answered_from_general_knowledge

                # Calibrate confidence score if local vector selection yielded 0
                if context.confidence_score <= 0.0:
                    if context.guardrails.web_fallback_used:
                        context.confidence_score = 0.75
                    elif gen_res.answered_from_general_knowledge:
                        context.confidence_score = 0.70
                    else:
                        context.confidence_score = 0.50

            t["extra"] = {
                "needs_web_search": context.needs_web_search,
                "from_general_knowledge": context.guardrails.answered_from_general_knowledge,
                "citations_claimed": len(context.citations),
                "answer_length": len(context.final_answer or ""),
            }
        except Exception as e:
            print(f"[Step 11] Structured generation error: {e}")
            context.final_answer = "Error generating or parsing response from model."
            context.citations = []
            context.needs_web_search = False
            t["extra"] = {"parse_error": str(e)}

        t["output_count"] = 1 if (context.final_answer and not context.is_abstained) else 0

    return context

