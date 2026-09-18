import json
from core.schemas import QueryContext
from core.interfaces import Generator

def execute(context: QueryContext, generator: Generator) -> QueryContext:
    """
    Step 11: Generation
    If not abstained, prompts the LLM to write the final answer based strictly
    on the validated evidence, and enforces strict citation formatting.
    """
    if context.is_abstained:
        context.final_answer = f"ABSTAIN: {context.abstention_reason}"
        context.citations = []
        return context
        
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

You must answer the question using ONLY the evidence provided below.
When you make a claim, you MUST cite the Evidence ID in brackets, like this: [123e4567-e89b-12d3-a456-426614174000].

EVIDENCE:
{evidence_text}

Return a JSON object with the following exact keys:
1. "content": The text of your final answer, including bracketed citations.
2. "citations": A list of strings containing all the Evidence IDs you cited.

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
    except Exception as e:
        context.final_answer = "Error parsing LLM generation output."
        context.citations = []
        
    return context
