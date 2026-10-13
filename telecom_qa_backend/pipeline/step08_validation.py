import re
from core.schemas import QueryContext, EvidenceValidationResult
from core.interfaces import Generator
from core.tracer import trace_step
from core.structured_llm import generate_structured

def execute(context: QueryContext, generator: Generator) -> QueryContext:
    """
    Step 8: Evidence Validation (LLM-as-a-Judge)
    Evaluates whether the retrieved technical chunks provide sufficient,
    consistent context to address the query prior to synthesis.
    Enforces Pydantic schema validation with automatic retry.
    """
    input_count = len(context.selected_evidence)

    with trace_step(context, "Step 08: Validation (LLM-as-Judge)", input_count=input_count) as t:
        if not context.selected_evidence:
            context.evidence_validation = EvidenceValidationResult(
                is_sufficient=False,
                contradiction_found=False,
                reasoning="No evidence was retrieved from the database."
            )
            t["output_count"] = 0
            t["extra"] = {"is_sufficient": False, "reason": "no_evidence"}
            return context

        query = context.understanding.normalized_query if context.understanding else context.original_query

        # Format the evidence clearly for the LLM
        evidence_text = ""
        for chunk in context.selected_evidence:
            meta = chunk.metadata
            evidence_text += f"\n--- CHUNK ID: {meta.chunk_id} | RELEASE: {meta.release_version} ---\n"
            evidence_text += f"{chunk.text}\n"

        prompt = f"""You are an expert telecom validation judge.
A user asked: "{query}"

You have retrieved the following 3GPP excerpts:
{evidence_text}

Task:
1. Determine if this evidence is sufficient to definitively answer the user's question, OR if it contains enough high-quality partial information to provide a highly useful, factual answer without guessing or hallucinating.
2. Check if there are any contradictions between the chunks (e.g. Rel-17 says X, but Rel-18 says Y).

Return ONLY a raw JSON object matching the following structure:
{{
  "is_sufficient": boolean,
  "contradiction_found": boolean,
  "contradiction_details": string or null,
  "reasoning": string
}}
Output strictly valid JSON.
"""

        try:
            val_result = generate_structured(
                generator=generator,
                prompt=prompt,
                response_model=EvidenceValidationResult,
                max_retries=2,
            )
            context.evidence_validation = val_result
            t["extra"] = {
                "is_sufficient": val_result.is_sufficient,
                "contradiction_found": val_result.contradiction_found,
            }
        except Exception as e:
            print(f"[Step 8] Structured validation fallback: {e}")
            context.evidence_validation = EvidenceValidationResult(
                is_sufficient=False,
                contradiction_found=False,
                reasoning="Validation parser failed. Safely assuming insufficient."
            )
            t["extra"] = {"parse_error": str(e), "is_sufficient": False}

        t["output_count"] = 1 if context.evidence_validation.is_sufficient else 0

    return context

