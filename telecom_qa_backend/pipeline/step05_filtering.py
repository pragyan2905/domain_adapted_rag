from typing import Set
from core.schemas import QueryContext

MIN_FUSION_SCORE = 0.005  # RRF scores are typically in range [0.005, 0.03]

def execute(context: QueryContext) -> QueryContext:
    """
    Step 5: Filtering and Deduplication
    1. Deduplicates by chunk_id (safety net, RRF should already handle this).
    2. Applies a minimum fusion score threshold to prune low-confidence candidates
       before sending them to the expensive cross-encoder reranker.
    """
    seen_ids: Set[str] = set()
    pruned_count = 0

    for chunk in context.fused_candidates:
        cid = chunk.metadata.chunk_id
        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        # Prune candidates that scored extremely low in fusion
        if chunk.score < MIN_FUSION_SCORE:
            pruned_count += 1
            continue

        context.filtered_candidates.append(chunk)

    if pruned_count:
        print(f"[Step 5] Pruned {pruned_count} low-confidence candidates (score < {MIN_FUSION_SCORE}).")

    return context
