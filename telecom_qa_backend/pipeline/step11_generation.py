import json
from core.schemas import QueryContext
from core.interfaces import Generator

def execute(context: QueryContext, generator: Generator) -> QueryContext:
    """
    Step 11: Generation
    If not abstained, prompts the LLM to write the final answer based strictly
    on the validated evidence, and enforces strict citation formatting.
    """
    query = context.understanding.normalized_query if context.understanding else context.original_query
    
    # Format evidence for the generator
    evidence_text = ""
    for chunk in context.selected_evidence:
        meta = chunk.metadata
        evidence_text += f"--- EVIDENCE ID: {meta.chunk_id} ---\n"
        evidence_text += f"{chunk.text}\n\n"
        
    prompt = f"""
You are an expert telecom 3GPP standards assistant.
A user asked: "{query}"

You have been provided with retrieved evidence from a local database:
{evidence_text}

Task:
1. Attempt to answer the user's question using ONLY the provided evidence. If you do this, you MUST cite the Evidence ID in brackets, like this: [123e4567-e89b-12d3-a456-426614174000]. Do NOT combine brackets (e.g. do not do [UUID1, UUID2]), and never use simple numbers like [1].
2. If the provided evidence is NOT sufficient to fully answer the question, try to answer the question using your own broad general knowledge of telecommunications. If you do this, DO NOT use any citations.
3. If the evidence is insufficient AND you are not completely confident in your general knowledge to answer the question accurately, you must set the "needs_web_search" flag to true and leave the content blank.

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
    except Exception as e:
        context.final_answer = "Error parsing LLM generation output."
        context.citations = []
        context.needs_web_search = False
        
    return context
