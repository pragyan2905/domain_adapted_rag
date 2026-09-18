from core.schemas import QueryContext

def execute(context: QueryContext, top_k: int = 5, max_context_tokens: int = 4000, score_threshold: float = -2.0) -> QueryContext:
    """
    Step 7: Evidence Selection
    Picks the absolute best chunks to pass to the LLM.
    Enforces a strict context window budget (tokens) and drops chunks 
    that score too low on the reranker to prevent polluting the context with garbage.
    """
    current_tokens = 0
    
    for chunk in context.reranked_candidates:
        # Stop if we hit the limit of chunks
        if len(context.selected_evidence) >= top_k:
            break
            
        # Stop if this chunk is mathematically irrelevant (BGE uses logit scores; negative scores mean bad match)
        if chunk.score < score_threshold:
            continue
            
        # Stop if adding this chunk blows our LLM context budget
        if current_tokens + chunk.metadata.token_count > max_context_tokens:
            continue
            
        context.selected_evidence.append(chunk)
        current_tokens += chunk.metadata.token_count
        
    return context
