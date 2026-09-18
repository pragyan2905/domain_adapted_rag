import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Core schemas and services
from core.schemas import QueryContext, GuardrailFlags
from services.qdrant_client import QdrantStore
from services.embed_bge import BGEEmbedder
from services.rerank_bge import BGEReranker
from services.llm_gemini import GeminiGenerator
from services.search_duckduckgo import DuckDuckGoSearch

# Pipeline Steps
from pipeline import (
    step01_understanding, step02_representation, step03_retrieval, step04_fusion,
    step05_filtering, step06_reranking, step07_selection, step08_validation,
    step09_web_fallback, step10_abstention, step11_generation, step12_guardrails
)

# -------------------------------------------------------------------------
# API Models
# -------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    is_abstained: bool
    abstention_reason: Optional[str]
    guardrails: GuardrailFlags
    # In a real app, you might also return the list of citation metadata (titles, URLs)

# -------------------------------------------------------------------------
# Application State
# -------------------------------------------------------------------------
app = FastAPI(title="3GPP Telecom QA Backend")

# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# We use a global dictionary or lifespan event to hold our singletons.
# For simplicity in this structure, we instantiate them globally.
# Note: In production, load API keys securely via a .env file or secret manager.
API_KEY = os.environ.get("GEMINI_API_KEY", "missing_key")

print("Initializing ML Models and Services. This may take a moment to load into VRAM...")
try:
    embedder = BGEEmbedder(use_fp16=True)
    reranker = BGEReranker(use_fp16=True)
    qdrant = QdrantStore(host="localhost", port=6333)
    web_search = DuckDuckGoSearch()
    generator = GeminiGenerator(api_key=API_KEY)
    print("All services initialized successfully.")
except Exception as e:
    print(f"Warning: Service initialization failed (Expected if models/DB aren't running yet): {e}")
    embedder, reranker, qdrant, web_search, generator = None, None, None, None, None

# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not generator or not embedder:
        raise HTTPException(status_code=500, detail="Backend ML services are not running.")
        
    # Instantiate the blank chassis
    ctx = QueryContext(original_query=request.query)
    
    # Run the 12-step assembly line
    try:
        ctx = step01_understanding.execute(ctx, generator)
        ctx = step02_representation.execute(ctx, embedder)
        ctx = step03_retrieval.execute(ctx, qdrant)
        ctx = step04_fusion.execute(ctx)
        ctx = step05_filtering.execute(ctx)
        ctx = step06_reranking.execute(ctx, reranker)
        ctx = step07_selection.execute(ctx)
        ctx = step08_validation.execute(ctx, generator)
        ctx = step09_web_fallback.execute(ctx, web_search, generator)
        ctx = step10_abstention.execute(ctx)
        ctx = step11_generation.execute(ctx, generator)
        ctx = step12_guardrails.execute(ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed at execution: {str(e)}")
        
    return QueryResponse(
        answer=ctx.final_answer or "No answer generated.",
        citations=ctx.citations,
        is_abstained=ctx.is_abstained,
        abstention_reason=ctx.abstention_reason,
        guardrails=ctx.guardrails
    )
