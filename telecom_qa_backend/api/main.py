import os
import json
import time
import uuid
import logging
from typing import List, Optional, AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("telecom_rag")

# Core schemas and services
from core.schemas import (
    QueryContext, GuardrailFlags, PipelineTrace, ConversationTurn,
    QueryRequest, QueryResponse,
    CompareResponse, CompareConfigResult
)
from services.qdrant_client import QdrantStore
from services.embed_bge import BGEEmbedder
from services.rerank_bge import BGEReranker
from services.llm_gemini import GeminiGenerator
from services.search_duckduckgo import DuckDuckGoSearch
from services.redis_cache import SemanticCache
from services.rate_limiter import QuotaLimiter
from services.database import DatabaseManager

# Pipeline Orchestration
from pipeline.orchestrator import PipelineOrchestrator

# -------------------------------------------------------------------------
# Application State
# -------------------------------------------------------------------------
app = FastAPI(
    title="3GPP Telecom QA Backend",
    description="Domain-adapted RAG system for 3GPP telecommunications standards.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GEMINI_API_KEY", "missing_key")
db_manager = DatabaseManager()

logger.info("Initializing ML Models and Services. This may take a moment to load into VRAM...")
try:
    embedder = BGEEmbedder(use_fp16=True)
    reranker = BGEReranker(use_fp16=True)
    qdrant = QdrantStore(host="localhost", port=6333)
    web_search = DuckDuckGoSearch()
    generator = GeminiGenerator(api_key=API_KEY)
    redis_cache = SemanticCache(host="localhost", port=6380)
    orchestrator = PipelineOrchestrator(
        embedder=embedder,
        reranker=reranker,
        qdrant=qdrant,
        generator=generator,
        web_search=web_search,
    )
    limiter = QuotaLimiter(redis_client=redis_cache.client if redis_cache else None)
    logger.info("All services, orchestrator, database, and rate limiter initialized successfully.")
except Exception as e:
    logger.error(f"Service initialization failed: {e}")
    embedder, reranker, qdrant, web_search, generator, redis_cache, orchestrator, limiter = None, None, None, None, None, None, None, None



# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def _load_conversation_history(session_id: str) -> List[ConversationTurn]:
    """Load last N turns from Redis for a session."""
    if not redis_cache or not session_id:
        return []
    try:
        key = f"session:{session_id}"
        raw = redis_cache.client.get(key)
        if raw:
            data = json.loads(raw)
            return [ConversationTurn(**t) for t in data]
    except Exception as e:
        logger.error(f"[Session] Failed to load history: {e}")
    return []


def _save_conversation_turn(session_id: str, question: str, answer: str):
    """Append this Q&A turn to Redis session history (TTL: 1 hour)."""
    if not redis_cache or not session_id:
        return
    try:
        key = f"session:{session_id}"
        raw = redis_cache.client.get(key)
        history = json.loads(raw) if raw else []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer[:500]})  # cap at 500 chars
        history = history[-10:]  # keep last 5 turns (10 messages)
        redis_cache.client.setex(key, 3600, json.dumps(history))
    except Exception as e:
        logger.error(f"[Session] Failed to save history: {e}")


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------

@app.get("/health")
async def health():
    """System health check endpoint."""
    status = {
        "status": "healthy",
        "services": {
            "embedder": embedder is not None,
            "reranker": reranker is not None,
            "generator": generator is not None,
            "orchestrator": orchestrator is not None,
        }
    }
    try:
        col = qdrant.client.get_collection(qdrant.collection_name)
        status["services"]["qdrant"] = True
        status["qdrant_chunks"] = col.points_count
    except Exception:
        status["services"]["qdrant"] = False

    try:
        redis_cache.client.ping()
        status["services"]["redis"] = True
        status["redis_cache_entries"] = redis_cache.client.dbsize()
    except Exception:
        status["services"]["redis"] = False

    all_healthy = all(status["services"].values())
    if not all_healthy:
        status["status"] = "degraded"

    return status


@app.get("/analytics")
async def analytics_endpoint():
    """System analytics and query metrics summary from SQLite audit database."""
    return db_manager.get_metrics_summary()


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, http_req: Request):
    """Standard blocking query endpoint with rate and quota limits."""
    if not generator or not embedder or not orchestrator:
        raise HTTPException(status_code=500, detail="Backend ML services are not running.")

    # Free-tier quota & rate limit check
    client_ip = http_req.headers.get("X-Forwarded-For", http_req.client.host if http_req.client else "127.0.0.1").split(",")[0].strip()
    if limiter:
        allowed, reason = limiter.check_and_increment(client_ip)
        if not allowed:
            raise HTTPException(status_code=429, detail=reason)

    session_id = request.session_id or str(uuid.uuid4())
    pipeline_start = time.perf_counter()

    # Step 0: Semantic Cache Intercept
    query_vector = None
    try:
        query_dense, _ = embedder.embed([request.query])
        query_vector = query_dense[0]

        if redis_cache:
            cached_data = redis_cache.check_cache(query_vector, threshold=0.90)
            if cached_data:
                cached_data["session_id"] = session_id
                if "pipeline_trace" not in cached_data:
                    cached_data["pipeline_trace"] = {"cache_hit": True, "total_duration_ms": 0, "steps": []}
                cached_data["pipeline_trace"]["cache_hit"] = True
                
                # Log cache hit
                db_manager.log_query(
                    session_id=session_id,
                    query=request.query,
                    answer=cached_data.get("answer", ""),
                    is_abstained=cached_data.get("is_abstained", False),
                    abstention_reason=cached_data.get("abstention_reason"),
                    confidence_score=cached_data.get("confidence_score", 0.0),
                    cache_hit=True,
                    latency_ms=round((time.perf_counter() - pipeline_start) * 1000, 2),
                    citations=cached_data.get("citations", []),
                    client_ip=client_ip,
                )
                return QueryResponse(**cached_data)
    except Exception as e:
        logger.warning(f"Cache check failed: {e}")

    # Load conversation history
    history = _load_conversation_history(session_id)

    # Validate session turn cap
    if limiter and len(history) // 2 >= limiter.max_turns_per_session:
        raise HTTPException(
            status_code=429,
            detail=f"Session turn cap reached ({limiter.max_turns_per_session} turns). Please start a New Session."
        )

    # Instantiate the blank context
    ctx = QueryContext(
        original_query=request.query,
        session_id=session_id,
        conversation_history=history,
    )

    try:
        ctx = orchestrator.run(ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    # Finalize trace
    ctx.pipeline_trace.total_duration_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

    response = QueryResponse(
        answer=ctx.final_answer or "No answer generated.",
        citations=ctx.citations,
        is_abstained=ctx.is_abstained,
        abstention_reason=ctx.abstention_reason,
        guardrails=ctx.guardrails,
        confidence_score=ctx.confidence_score,
        session_id=session_id,
        pipeline_trace=ctx.pipeline_trace,
    )

    # Save to Semantic Cache
    if redis_cache and query_vector and not ctx.is_abstained:
        ttl = 2592000
        if ctx.guardrails.answered_from_general_knowledge or ctx.guardrails.web_fallback_used:
            ttl = 86400
        try:
            redis_cache.save_to_cache(query_vector, response.model_dump_json(), ttl_seconds=ttl)
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")

    # Save conversation turn
    if ctx.final_answer and not ctx.is_abstained:
        _save_conversation_turn(session_id, request.query, ctx.final_answer)

    # Log query execution to SQLite audit database
    db_manager.log_query(
        session_id=session_id,
        query=request.query,
        answer=response.answer,
        is_abstained=response.is_abstained,
        abstention_reason=response.abstention_reason,
        confidence_score=response.confidence_score,
        cache_hit=False,
        latency_ms=ctx.pipeline_trace.total_duration_ms,
        citations=response.citations,
        client_ip=client_ip,
    )

    return response


@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest, http_req: Request):
    """
    Streaming SSE endpoint with free-tier rate & session limiting.
    Emits one JSON event per pipeline step as it completes.
    """
    if not generator or not embedder or not orchestrator:
        raise HTTPException(status_code=500, detail="Backend ML services are not running.")

    # Free-tier quota & rate limit check
    client_ip = http_req.headers.get("X-Forwarded-For", http_req.client.host if http_req.client else "127.0.0.1").split(",")[0].strip()
    if limiter:
        allowed, reason = limiter.check_and_increment(client_ip)
        if not allowed:
            raise HTTPException(status_code=429, detail=reason)

    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator[str, None]:
        def sse(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        yield sse({"type": "start", "session_id": session_id, "message": "Pipeline started..."})

        pipeline_start = time.perf_counter()

        # Step 0: Cache check
        query_vector = None
        try:
            query_dense, _ = embedder.embed([request.query])
            query_vector = query_dense[0]
            if redis_cache:
                cached_data = redis_cache.check_cache(query_vector, threshold=0.90)
                if cached_data:
                    cached_data["session_id"] = session_id
                    if "pipeline_trace" not in cached_data:
                        cached_data["pipeline_trace"] = {"cache_hit": True, "total_duration_ms": 0.0, "steps": []}
                    cached_data["pipeline_trace"]["cache_hit"] = True

                    db_manager.log_query(
                        session_id=session_id,
                        query=request.query,
                        answer=cached_data.get("answer", ""),
                        is_abstained=cached_data.get("is_abstained", False),
                        abstention_reason=cached_data.get("abstention_reason"),
                        confidence_score=cached_data.get("confidence_score", 0.0),
                        cache_hit=True,
                        latency_ms=round((time.perf_counter() - pipeline_start) * 1000, 2),
                        citations=cached_data.get("citations", []),
                        client_ip=client_ip,
                    )

                    yield sse({"type": "cache_hit", "message": "Semantic cache hit! Returning cached response."})
                    yield sse({"type": "complete", "result": cached_data})
                    return
        except Exception as e:
            logger.warning(f"Cache check failed in stream: {e}")

        history = _load_conversation_history(session_id)
        ctx = QueryContext(
            original_query=request.query,
            session_id=session_id,
            conversation_history=history,
        )

        try:
            async for event in orchestrator.stream_execution(ctx):
                yield sse(event)
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})
            return

        ctx.pipeline_trace.total_duration_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

        result = QueryResponse(
            answer=ctx.final_answer or "No answer generated.",
            citations=ctx.citations,
            is_abstained=ctx.is_abstained,
            abstention_reason=ctx.abstention_reason,
            guardrails=ctx.guardrails,
            confidence_score=ctx.confidence_score,
            session_id=session_id,
            pipeline_trace=ctx.pipeline_trace,
        )

        # Save to cache + session
        if redis_cache and query_vector and not ctx.is_abstained:
            ttl = 2592000 if not (ctx.guardrails.answered_from_general_knowledge or ctx.guardrails.web_fallback_used) else 86400
            try:
                redis_cache.save_to_cache(query_vector, result.model_dump_json(), ttl_seconds=ttl)
            except Exception as e:
                logger.error(f"Error saving to cache: {e}")

        if ctx.final_answer and not ctx.is_abstained:
            _save_conversation_turn(session_id, request.query, ctx.final_answer)

        # Log to SQLite audit database
        db_manager.log_query(
            session_id=session_id,
            query=request.query,
            answer=result.answer,
            is_abstained=result.is_abstained,
            abstention_reason=result.abstention_reason,
            confidence_score=result.confidence_score,
            cache_hit=False,
            latency_ms=ctx.pipeline_trace.total_duration_ms,
            citations=result.citations,
            client_ip=client_ip,
        )

        yield sse({"type": "complete", "result": result.model_dump()})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/query/compare", response_model=CompareResponse)
async def query_compare_endpoint(request: QueryRequest, http_req: Request):
    """
    Ablation Study / A-B Comparison Endpoint:
    Executes the query against two pipeline configurations:
      1. Advanced 12-Step RAG (HyDE + RRF + BGE Cross-Encoder Rerank + Guardrails)
      2. Baseline Naive RAG (Dense vector search only, no HyDE, top-5 chunks, direct generation, no guardrails)
    Returns side-by-side performance metrics and answers.
    """
    if not generator or not embedder or not orchestrator or not qdrant:
        raise HTTPException(status_code=500, detail="Backend ML services are not running.")

    client_ip = http_req.headers.get("X-Forwarded-For", http_req.client.host if http_req.client else "127.0.0.1").split(",")[0].strip()
    if limiter:
        allowed, reason = limiter.check_and_increment(client_ip)
        if not allowed:
            raise HTTPException(status_code=429, detail=reason)

    query = request.query
    session_id = request.session_id or str(uuid.uuid4())

    # --- Run Config A: Advanced 12-Step RAG ---
    start_adv = time.perf_counter()
    ctx_adv = QueryContext(original_query=query, session_id=session_id)
    ctx_adv = orchestrator.run(ctx_adv)
    dur_adv = round((time.perf_counter() - start_adv) * 1000, 2)

    advanced_result = CompareConfigResult(
        config_name="Advanced 12-Step RAG (HyDE + RRF + BGE Reranker + Guardrails)",
        answer=ctx_adv.final_answer or "No answer.",
        confidence_score=ctx_adv.confidence_score,
        citations_count=len(ctx_adv.citations),
        duration_ms=dur_adv,
        retrieval_count=len(ctx_adv.selected_evidence),
        guardrails_passed=not ctx_adv.guardrails.unresolved_citations and not ctx_adv.guardrails.ungrounded_numbers,
    )

    # --- Run Config B: Baseline Naive RAG ---
    start_naive = time.perf_counter()
    query_dense, _ = embedder.embed([query])
    raw_chunks = qdrant.search_dense(query_dense[0], limit=5)

    context_text = "\n\n".join([f"[{c.metadata.chunk_id}]: {c.text}" for c in raw_chunks])
    naive_prompt = f"Answer the question based on this context:\n{context_text}\n\nQuestion: {query}"
    naive_res = generator.generate(naive_prompt, context=raw_chunks)
    dur_naive = round((time.perf_counter() - start_naive) * 1000, 2)

    naive_result = CompareConfigResult(
        config_name="Baseline Naive RAG (Dense Top-5, No HyDE, No Reranker)",
        answer=naive_res.content[:500] + ("..." if len(naive_res.content) > 500 else ""),
        confidence_score=round(raw_chunks[0].score if raw_chunks else 0.0, 3),
        citations_count=0,
        duration_ms=dur_naive,
        retrieval_count=len(raw_chunks),
        guardrails_passed=False,
    )

    return CompareResponse(
        query=query,
        advanced_rag=advanced_result,
        baseline_naive_rag=naive_result,
    )

