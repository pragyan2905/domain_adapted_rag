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
        # Format evidence for the generator
        evidence_text = ""
        for chunk in context.selected_evidence:
            meta = chunk.metadata
            evidence_text += f"--- EVIDENCE ID: {meta.chunk_id} ---\n"
            evidence_text += f"{chunk.text}\n\n"

        # Inject conversation history for multi-turn awareness
        history_text = ""
        if context.conversation_history:
            history_text = "\n\nPREVIOUS CONVERSATION CONTEXT:\n"
            for turn in context.conversation_history[-4:]:  # last 4 turns max
                history_text += f"{turn.role.upper()}: {turn.content}\n"

        prompt = f"""You are an expert telecom 3GPP standards assistant.
{history_text}
A user asked: "{query}"

You have been provided with retrieved evidence from a local database:
{evidence_text}

Task:
1. Attempt to answer the user's question using ONLY the provided evidence. If you do this, you MUST cite the Evidence ID in brackets, like this: [123e4567-e89b-12d3-a456-426614174000]. Do NOT combine brackets (e.g. do not do [UUID1, UUID2]), and never use simple numbers like [1].
2. If the provided evidence is NOT sufficient to fully answer the question, try to answer using your own broad general knowledge of telecommunications. If you do this, DO NOT use any citations.
3. If the evidence is insufficient AND you are not completely confident in your general knowledge, set "needs_web_search" to true and leave content blank.

Return a JSON object matching this schema:
{{
  "content": "The text of your final answer (or blank if needs web search)",
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
            context.final_answer = gen_res.content or "No answer could be formulated."
            context.citations = gen_res.citations
            context.needs_web_search = gen_res.needs_web_search
            context.guardrails.answered_from_general_knowledge = gen_res.answered_from_general_knowledge
            t["extra"] = {
                "needs_web_search": context.needs_web_search,
                "from_general_knowledge": context.guardrails.answered_from_general_knowledge,
                "citations_claimed": len(context.citations),
                "answer_length": len(context.final_answer),
            }
        except Exception as e:
            print(f"[Step 11] Structured generation error: {e}")
            context.final_answer = "Error generating or parsing response from model."
            context.citations = []
            context.needs_web_search = False
            t["extra"] = {"parse_error": str(e)}

        t["output_count"] = 1 if context.final_answer else 0

    return context

