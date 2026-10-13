import uuid
from core.schemas import QueryContext, RetrievedChunk, ChunkMetadata, SourceType
from core.interfaces import WebSearch
from core.tracer import trace_step

def execute(context: QueryContext, web_search: WebSearch) -> QueryContext:
    """
    Step 9: Web Fallback (Conditional Branch)
    This step ONLY runs if the evidence validation step flagged insufficient local evidence.
    """
    query = context.understanding.normalized_query if context.understanding else context.original_query

    with trace_step(context, "Step 09: Web Fallback", input_count=0) as t:
        web_results = web_search.search(query=query)

        if not web_results:
            t["extra"] = {"web_results": 0, "triggered": True}
            return context

        context.guardrails.web_fallback_used = True

        for res in web_results:
            meta = ChunkMetadata(
                chunk_id=str(uuid.uuid4()),
                ts_number="Web",
                series="Web",
                release_version="Unknown",
                clause_id=res.url,
                clause_title=res.title,
                breadcrumb="Web Search Result",
                token_count=len(res.content) // 4,
                source_type=SourceType.WEB
            )
            web_chunk = RetrievedChunk(
                text=f"SOURCE: {res.url}\n\n{res.content}",
                metadata=meta,
                score=1.0
            )
            context.selected_evidence.append(web_chunk)
            if res.url not in context.citations:
                context.citations.append(res.url)

        t["output_count"] = len(web_results)
        t["extra"] = {"web_results": len(web_results), "triggered": True}

    return context
