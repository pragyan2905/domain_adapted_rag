import json
import re
from core.schemas import QueryContext, EvidenceValidationResult
from core.interfaces import Generator

def execute(context: QueryContext, generator: Generator) -> QueryContext:
    """
    Step 8: Evidence Validation (LLM-as-a-judge)
    Before hallucinating an answer, we force the LLM to inspect the retrieved
    chunks and declare if they are actually sufficient to answer the question.
    """
    if not context.selected_evidence:
        context.evidence_validation = EvidenceValidationResult(
            is_sufficient=False,
            contradiction_found=False,
            reasoning="No evidence was retrieved from the database."
        )
        return context
        
    query = context.understanding.normalized_query if context.understanding else context.original_query
    
    # Format the evidence clearly for the LLM
    evidence_text = ""
    for chunk in context.selected_evidence:
        meta = chunk.metadata
        evidence_text += f"\n--- CHUNK ID: {meta.chunk_id} | RELEASE: {meta.release_version} ---\n"
        evidence_text += f"{chunk.text}\n"
        
    prompt = f"""
 You are an expert telecom validation judge.
 A user asked: "{query}"
 
 You have retrieved the following 3GPP excerpts:
 {evidence_text}
 
 Task:
 1. Determine if this evidence is sufficient to definitively answer the user's question, OR if it contains enough high-quality partial information to provide a highly useful, factual answer without guessing or hallucinating.
 2. Check if there are any contradictions between the chunks (e.g. Rel-17 says X, but Rel-18 says Y).
 
 Return ONLY a raw JSON object — no markdown, no code fences, no explanation. Just the JSON.
 The JSON must have exactly these keys:
 - "is_sufficient": boolean
 - "contradiction_found": boolean
 - "contradiction_details": string or null
 - "reasoning": string
"""

    result = generator.generate(prompt=prompt, context=[])
    raw_content = result.content.strip()
    
    # Robustly extract the first JSON object from any LLM output.
    # This handles: raw JSON, ```json fences, extra prose before/after, etc.
    json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
    
    try:
        if not json_match:
            raise ValueError(f"No JSON object found in LLM response: {raw_content!r}")
        data = json.loads(json_match.group())
        context.evidence_validation = EvidenceValidationResult(**data)
    except Exception as e:
        print(f"[Step 8] Validation parser error: {e}")
        print(f"[Step 8] Raw LLM output was: {raw_content!r}")
        # If the judge fails, fail safe (assume insufficient) to prevent hallucinations
        context.evidence_validation = EvidenceValidationResult(
            is_sufficient=False,
            contradiction_found=False,
            reasoning="Validation parser failed. Safely assuming insufficient."
        )
        
    return context
