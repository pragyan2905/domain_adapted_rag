import re
from core.schemas import QueryContext
from services.embed_bge import BGEEmbedder

def execute(context: QueryContext, embedder: BGEEmbedder) -> QueryContext:
    """
    Step 2: Query Representation
    Converts the normalized query into dense and sparse vectors.
    Also applies regex heuristics to extract strict metadata filters (e.g. TS 38.331).
    """
    query = context.understanding.normalized_query if context.understanding else context.original_query
    
    # 1. Embed the query (dense + sparse simultaneously via BGE-M3)
    dense_vecs, sparse_vecs = embedder.embed([query])
    
    if dense_vecs and sparse_vecs:
        context.query_dense_vector = dense_vecs[0]
        context.query_sparse_vector = sparse_vecs[0]
        
    # 2. Extract structured filters (e.g., TS number or Release version)
    # Simple regex rules to pull out 3GPP specifiers to hard-filter the vector DB
    ts_pattern = re.compile(r"TS\s*(\d{2}\.\d{3})", re.IGNORECASE)
    rel_pattern = re.compile(r"(?:Rel|Release)[-\s]*(\d{2})", re.IGNORECASE)
    
    ts_matches = ts_pattern.findall(context.original_query)
    if ts_matches:
        context.target_ts_numbers = list(set(ts_matches))
        
    rel_matches = rel_pattern.findall(context.original_query)
    if rel_matches:
        context.target_releases = [f"Rel-{m}" for m in set(rel_matches)]
        
    return context
