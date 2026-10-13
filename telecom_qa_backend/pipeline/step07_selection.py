from core.schemas import QueryContext
from core.tracer import trace_step

def execute(context: QueryContext, top_k: int = 5, max_context_tokens: int = 4000, score_threshold: float = -2.0) -> QueryContext:
    """
    Step 7: Evidence Selection
    Picks the absolute best chunks to pass to the LLM.
    Enforces a strict context window budget (tokens) and drops chunks
    that score too low on the reranker to prevent polluting the context.

    Also computes a calibrated confidence_score based on reranker score distribution.
    """
    input_count = len(context.reranked_candidates)

    with trace_step(context, "Step 07: Selection", input_count=input_count) as t:
        current_tokens = 0

        for chunk in context.reranked_candidates:
            if len(context.selected_evidence) >= top_k:
                break
            if chunk.score < score_threshold:
                continue
            if current_tokens + chunk.metadata.token_count > max_context_tokens:
                continue

            context.selected_evidence.append(chunk)
            current_tokens += chunk.metadata.token_count

        # --- Confidence Score Calculation ---
        # Based on reranker score distribution of selected evidence.
        # BGE reranker scores are logits — higher is better. Typical good match: 3-8, poor: <0
        if context.selected_evidence:
            scores = [c.score for c in context.selected_evidence]
            avg_score = sum(scores) / len(scores)
            # Sigmoid-like normalization to [0, 1]: score of 0 → 0.5, score of 5 → 0.99
            import math
            context.confidence_score = round(1 / (1 + math.exp(-avg_score * 0.5)), 3)
        else:
            context.confidence_score = 0.0

        t["output_count"] = len(context.selected_evidence)
        t["extra"] = {
            "tokens_used": current_tokens,
            "confidence_score": context.confidence_score,
            "evidence_ts_numbers": list({c.metadata.ts_number for c in context.selected_evidence}),
            "evidence_releases": list({c.metadata.release_version for c in context.selected_evidence}),
        }

    return context
