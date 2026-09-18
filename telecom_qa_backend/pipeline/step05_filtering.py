from typing import Set
from core.schemas import QueryContext

def execute(context: QueryContext) -> QueryContext:
    """
    Step 5: Filtering and Deduplication
    Removes duplicate chunks that may have been retrieved by both
    the dense and sparse vectors before we send them to the heavy cross-encoder.
    """
    seen_ids: Set[str] = set()
    
    for chunk in context.fused_candidates:
        cid = chunk.metadata.chunk_id
        if cid not in seen_ids:
            seen_ids.add(cid)
            context.filtered_candidates.append(chunk)
            
    return context
