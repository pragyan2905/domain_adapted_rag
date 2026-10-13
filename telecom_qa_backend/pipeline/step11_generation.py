import json
from core.schemas import QueryContext
from core.interfaces import Generator
from core.tracer import trace_step

def execute(context: QueryContext, generator: Generator) -> QueryContext:
    """
    Step 11: Generation
    If not abstained, prompts the LLM to write the final answer based strictly
    on the validated evidence, and enforces strict citation formatting.
    Injects conversation history for multi-turn context awareness.
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

        prompt = f"""
You are an expert telecom 3GPP standards assistant.
{history_text}
A user asked: "{query}"

You have been provided with retrieved evidence from a local database:
{evidence_text}

Task:
1. Attempt to answer the user's question using ONLY the provided evidence. If you do this, you MUST cite the Evidence ID in brackets, like this: [123e4567-e89b-12d3-a456-426614174000]. Do NOT combine brackets (e.g. do not do [UUID1, UUID2]), and never use simple numbers like [1].
2. If the provided evidence is NOT sufficient to fully answer the question, try to answer using your own broad general knowledge of telecommunications. If you do this, DO NOT use any citations.
3. If the evidence is insufficient AND you are not completely confident in your general knowledge, set "needs_web_search" to true and leave content blank.

Return a JSON object with the following exact keys:
1. "content": The text of your final answer (if applicable).
2. "citations": A list of strings containing all the Evidence IDs you cited.
3. "needs_web_search": boolean. true ONLY if you cannot answer using evidence AND you are not confident in your general knowledge.
4. "answered_from_general_knowledge": boolean. true ONLY if you ignored the evidence and answered from memory.

Output strictly raw JSON.
"""

        result = generator.generate(prompt=prompt, context=context.selected_evidence)

        raw_content = result.content.strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]

        try:
            data = json.loads(raw_content.strip())
            context.final_answer = data.get("content", "Error generating response.")
            context.citations = data.get("citations", [])
            context.needs_web_search = data.get("needs_web_search", False)
            context.guardrails.answered_from_general_knowledge = data.get("answered_from_general_knowledge", False)
            t["extra"] = {
                "needs_web_search": context.needs_web_search,
                "from_general_knowledge": context.guardrails.answered_from_general_knowledge,
                "citations_claimed": len(context.citations),
                "answer_length": len(context.final_answer),
            }
        except Exception as e:
            context.final_answer = "Error parsing LLM generation output."
            context.citations = []
            context.needs_web_search = False
            t["extra"] = {"parse_error": str(e)}

        t["output_count"] = 1 if context.final_answer else 0

    return context
