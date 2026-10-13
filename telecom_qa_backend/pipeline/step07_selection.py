import math
from typing import List
from core.schemas import QueryContext, RetrievedChunk
from core.tracer import trace_step

def execute(
    context: QueryContext,
    top_k: int = 5,
    max_context_tokens: int = 4000,
    score_threshold: float = -2.0,
    hierarchical_expansion: bool = True
) -> QueryContext:
    """
    Step 7: Evidence Selection & Hierarchical Context Expansion
    Picks top candidate chunks within context token budgets.
    Applies hierarchical ordering (Parent-Child grouping via breadcrumbs and clause IDs)
    so the LLM receives logically structured procedural context instead of fragmented snippets.
    """
    input_count = len(context.reranked_candidates)

    with trace_step(context, "Step 07: Selection", input_count=input_count) as t:
        current_tokens = 0
        selected: List[RetrievedChunk] = []

        # 1. Primary Selection based on reranker scores
        for chunk in context.reranked_candidates:
            if len(selected) >= top_k:
                break
            if chunk.score < score_threshold:
                continue
            if current_tokens + chunk.metadata.token_count > max_context_tokens:
                continue

            selected.append(chunk)
            current_tokens += chunk.metadata.token_count

        # 2. Hierarchical Context Expansion (Parent-Child awareness)
        # Sort selected evidence by technical hierarchy: TS Number -> Clause ID -> Breadcrumb
        if hierarchical_expansion and len(selected) > 1:
            try:
                selected.sort(key=lambda c: (
                    c.metadata.ts_number or "",
                    c.metadata.parent_clause_id or "",
                    c.metadata.clause_id or "",
                ))
            except Exception:
                pass

        context.selected_evidence = selected

        # 3. Calibrated Confidence Score Calculation (Sigmoid over reranker logits)
        if context.selected_evidence:
            scores = [c.score for c in context.selected_evidence]
            avg_score = sum(scores) / len(scores)
            context.confidence_score = round(1 / (1 + math.exp(-avg_score * 0.5)), 3)
        else:
            context.confidence_score = 0.0

        t["output_count"] = len(context.selected_evidence)
        t["extra"] = {
            "tokens_used": current_tokens,
            "confidence_score": context.confidence_score,
            "evidence_ts_numbers": list({c.metadata.ts_number for c in context.selected_evidence}),
            "evidence_releases": list({c.metadata.release_version for c in context.selected_evidence}),
            "parent_clauses_linked": len({c.metadata.parent_clause_id for c in context.selected_evidence if c.metadata.parent_clause_id}),
        }

    return context
