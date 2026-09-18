from core.schemas import QueryContext
from services.rerank_bge import BGEReranker

def execute(context: QueryContext, reranker: BGEReranker) -> QueryContext:
    """
    Step 6: Cross-Encoder Reranking
    Scores the filtered candidates against the user's query using a Cross-Encoder.
    Cross-encoders are slower but much more accurate than the Bi-Encoders used in Step 3.
    """
    if not context.filtered_candidates:
        return context
        
    query = context.understanding.normalized_query if context.understanding else context.original_query
    passages = [chunk.text for chunk in context.filtered_candidates]
    
    # Compute the massive cross-attention scores
    scores = reranker.compute_scores(query=query, passages=passages)
    
    # Overwrite the fast RRF scores with the highly-accurate cross-encoder scores
    for i, chunk in enumerate(context.filtered_candidates):
        chunk.score = scores[i]
        
    # Sort the candidates by the new reranker score descending
    context.reranked_candidates = sorted(
        context.filtered_candidates, 
        key=lambda x: x.score, 
        reverse=True
    )
    
    return context
