import uuid
from core.schemas import QueryContext, RetrievedChunk, ChunkMetadata, SourceType
from core.interfaces import WebSearch, Generator

# We import the exact logic from Step 8 to re-validate
from pipeline.step08_validation import execute as run_validation

def execute(context: QueryContext, web_search: WebSearch, generator: Generator) -> QueryContext:
    """
    Step 9: Web Fallback (Conditional Branch)
    This step ONLY runs if Step 8 (Validation) explicitly flagged the 
    local Qdrant evidence as insufficient.
    """
    if context.evidence_validation and context.evidence_validation.is_sufficient:
        return context  # We have what we need, skip the web
        
    query = context.understanding.normalized_query if context.understanding else context.original_query
    
    # 1. Hit the Web Search API
    web_results = web_search.search(query=query)
    
    if not web_results:
        return context  # Web search failed or was disabled, just return
        
    context.guardrails.web_fallback_used = True
    
    # 2. Wrap web results into our strict pipeline format
    for res in web_results:
        meta = ChunkMetadata(
            chunk_id=str(uuid.uuid4()),
            ts_number="Web",
            series="Web",
            release_version="Unknown",
            clause_id=res.url, # Hack: use URL as citation target
            clause_title=res.title,
            breadcrumb="Web Search Result",
            token_count=len(res.content) // 4, # rough estimation
            source_type=SourceType.WEB
        )
        
        web_chunk = RetrievedChunk(
            text=f"SOURCE: {res.url}\n\n{res.content}",
            metadata=meta,
            score=1.0 # Force include it
        )
        
        context.selected_evidence.append(web_chunk)
        
    # 3. Re-run Step 8 Validation on the newly expanded evidence pool
    # We mutate context.evidence_validation with the new verdict
    context = run_validation(context, generator)
    
    return context
