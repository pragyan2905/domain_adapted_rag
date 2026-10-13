from core.schemas import QueryContext
from services.qdrant_client import QdrantStore
from core.tracer import trace_step

def execute(context: QueryContext, qdrant: QdrantStore, limit: int = 20) -> QueryContext:
    """
    Step 3: Hybrid Retrieval
    Queries Qdrant using the Dense vector and Sparse vector separately.
    Saves the two disjoint lists of candidates to the context for Step 4 to fuse.
    """
    if not context.query_dense_vector or not context.query_sparse_vector:
        return context

    with trace_step(context, "Step 03: Retrieval", input_count=0) as t:
        # 1. Search Dense (Semantic Match)
        context.dense_candidates = qdrant.search_dense(
            dense_vec=context.query_dense_vector,
            limit=limit,
            ts_filters=context.target_ts_numbers
        )

        # 2. Search Sparse (Lexical/BM25 Exact Keyword Match)
        context.sparse_candidates = qdrant.search_sparse(
            sparse_vec=context.query_sparse_vector,
            limit=limit,
            ts_filters=context.target_ts_numbers
        )

        total_retrieved = len(context.dense_candidates) + len(context.sparse_candidates)
        t["output_count"] = total_retrieved
        t["extra"] = {
            "dense_hits": len(context.dense_candidates),
            "sparse_hits": len(context.sparse_candidates),
        }

    return context
