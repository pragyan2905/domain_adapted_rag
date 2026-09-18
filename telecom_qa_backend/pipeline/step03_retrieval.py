from core.schemas import QueryContext
from services.qdrant_client import QdrantStore

def execute(context: QueryContext, qdrant: QdrantStore, limit: int = 20) -> QueryContext:
    """
    Step 3: Hybrid Retrieval
    Queries Qdrant using the Dense vector and Sparse vector separately.
    Saves the two disjoint lists of candidates to the context for Step 4 to fuse.
    """
    if not context.query_dense_vector or not context.query_sparse_vector:
        return context
        
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
    
    return context
