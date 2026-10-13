import os
import json
import time
import uuid
from typing import List, Optional, AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Core schemas and services
from core.schemas import (
    QueryContext, GuardrailFlags, PipelineTrace, ConversationTurn,
    QueryRequest, QueryResponse
)
from services.qdrant_client import QdrantStore
from services.embed_bge import BGEEmbedder
from services.rerank_bge import BGEReranker
from services.llm_gemini import GeminiGenerator
from services.search_duckduckgo import DuckDuckGoSearch
from services.redis_cache import SemanticCache
from services.rate_limiter import QuotaLimiter

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

print("Initializing ML Models and Services. This may take a moment to load into VRAM...")
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
    print("All services, orchestrator, and rate limiter initialized successfully.")
except Exception as e:
    print(f"Warning: Service initialization failed: {e}")
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
        print(f"[Session] Failed to load history: {e}")
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
        print(f"[Session] Failed to save history: {e}")


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
                return QueryResponse(**cached_data)
    except Exception as e:
        print(f"Cache check failed: {e}")

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
            print(f"Error saving to cache: {e}")

    # Save conversation turn
    if ctx.final_answer and not ctx.is_abstained:
        _save_conversation_turn(session_id, request.query, ctx.final_answer)

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
                    yield sse({"type": "cache_hit", "message": "Semantic cache hit! Returning cached response."})
                    yield sse({"type": "complete", "result": cached_data})
                    return
        except Exception as e:
            print(f"Cache check failed in stream: {e}")

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
            except Exception:
                pass

        if ctx.final_answer and not ctx.is_abstained:
            _save_conversation_turn(session_id, request.query, ctx.final_answer)

        yield sse({"type": "complete", "result": result.model_dump()})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
