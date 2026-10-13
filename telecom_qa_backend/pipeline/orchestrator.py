"""
pipeline/orchestrator.py
========================
Centralized orchestrator for the 12-step 3GPP RAG pipeline.

Encapsulates execution order, dependency injection, and SSE step streaming
so that api/main.py remains a clean, modular routing layer.
"""

import time
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional

from core.schemas import QueryContext
from core.interfaces import Generator, WebSearch
from services.qdrant_client import QdrantStore
from services.embed_bge import BGEEmbedder
from services.rerank_bge import BGEReranker

from pipeline import (
    step01_understanding,
    step02_representation,
    step03_retrieval,
    step04_fusion,
    step05_filtering,
    step06_reranking,
    step07_selection,
    step08_validation,
    step09_web_fallback,
    step10_abstention,
    step11_generation,
    step12_guardrails,
)


class PipelineOrchestrator:
    """
    Manages end-to-end execution of the 12-step RAG pipeline.
    Provides both synchronous batch execution and asynchronous SSE step streaming.
    """

    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        reranker: Optional[BGEReranker] = None,
        qdrant: Optional[QdrantStore] = None,
        generator: Optional[Generator] = None,
        web_search: Optional[WebSearch] = None,
    ):
        self.embedder = embedder
        self.reranker = reranker
        self.qdrant = qdrant
        self.generator = generator
        self.web_search = web_search

    def run(self, context: QueryContext) -> QueryContext:
        """
        Executes the full 12-step pipeline synchronously.

        Execution Flow:
          1. Understanding -> Acronym expansion & intent detection
          2. Representation -> Dense & sparse query vector generation
          3. Retrieval -> Hybrid search against Qdrant
          4. Fusion -> Reciprocal Rank Fusion (RRF)
          5. Filtering -> Score threshold pruning
          6. Reranking -> BGE Cross-Encoder scoring
          7. Selection -> Evidence selection & confidence calibration
          8. Validation -> LLM-as-judge evidence sufficiency check
          9. Web Fallback -> External search if evidence is insufficient
          10. Abstention Gate -> Hard refusal if evidence still insufficient
          11. Generation -> Domain-adapted LLM synthesis with citations
          12. Guardrails -> Post-generation factual & citation checks
        """
        # Step 01: Query Understanding
        context = step01_understanding.execute(context, self.generator)

        # Step 02: Query Representation (with HyDE)
        context = step02_representation.execute(context, self.embedder, generator=self.generator, enable_hyde=True)

        # Step 03: Hybrid Retrieval
        context = step03_retrieval.execute(context, self.qdrant)

        # Step 04: Candidate Fusion
        context = step04_fusion.execute(context)

        # Step 05: Relevance Filtering
        context = step05_filtering.execute(context)

        # Step 06: Cross-Encoder Reranking
        context = step06_reranking.execute(context, self.reranker)

        # Step 07: Evidence Selection
        context = step07_selection.execute(context)

        # Step 08: Evidence Validation (LLM-as-a-judge)
        context = step08_validation.execute(context, self.generator)

        # Step 09: Conditional Web Fallback
        if context.evidence_validation and not context.evidence_validation.is_sufficient:
            context = step09_web_fallback.execute(context, self.web_search)
            # Re-validate combined evidence
            context = step08_validation.execute(context, self.generator)

        # Step 10: Abstention Gate
        context = step10_abstention.execute(context)

        # Step 11: Generation (skipped internally if is_abstained is True)
        context = step11_generation.execute(context, self.generator)

        # Step 12: Deterministic Guardrails
        context = step12_guardrails.execute(context)

        return context

    async def stream_execution(self, context: QueryContext) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Asynchronously executes the pipeline step-by-step, yielding progress and
        completion event payloads suitable for Server-Sent Events (SSE).
        """
        steps = [
            ("Step 01: Understanding",        lambda c: step01_understanding.execute(c, self.generator),  "Normalizing acronyms, detecting intent"),
            ("Step 02: Representation",       lambda c: step02_representation.execute(c, self.embedder, generator=self.generator, enable_hyde=True),  "Generating dense + sparse embeddings with HyDE"),
            ("Step 03: Hybrid Retrieval",     lambda c: step03_retrieval.execute(c, self.qdrant),         "Querying vector database"),
            ("Step 04: RRF Fusion",           lambda c: step04_fusion.execute(c),                         "Fusing dense + sparse rankings"),
            ("Step 05: Filtering",            lambda c: step05_filtering.execute(c),                      "Pruning low-confidence candidates"),
            ("Step 06: Cross-Encoder Rerank", lambda c: step06_reranking.execute(c, self.reranker),      "Re-scoring with cross-encoder"),
            ("Step 07: Evidence Selection",   lambda c: step07_selection.execute(c),                      "Selecting top-K evidence chunks"),
            ("Step 08: Validation",           lambda c: step08_validation.execute(c, self.generator),    "LLM-as-judge: validating evidence"),
            ("Step 09: Web Fallback",         None,                                                       "Fetching supplementary web data"),
            ("Step 10: Abstention Gate",      lambda c: step10_abstention.execute(c),                     "Checking if abstention is needed"),
            ("Step 11: Generation",           lambda c: step11_generation.execute(c, self.generator),    "Generating final answer with LLM"),
            ("Step 12: Guardrails",           lambda c: step12_guardrails.execute(c),                     "Running post-generation guardrails"),
        ]

        for i, (step_name, step_fn, step_msg) in enumerate(steps):
            # Special branch: Web Fallback
            if step_name == "Step 09: Web Fallback":
                if context.evidence_validation and not context.evidence_validation.is_sufficient:
                    yield {"type": "step_progress", "step": step_name, "message": step_msg, "index": i}
                    context = await asyncio.to_thread(step09_web_fallback.execute, context, self.web_search)
                    context = await asyncio.to_thread(step08_validation.execute, context, self.generator)
                    trace = context.pipeline_trace.steps[-1].model_dump() if context.pipeline_trace.steps else {}
                    yield {"type": "step_done", "step": step_name, "trace": trace, "index": i}
                else:
                    yield {"type": "step_skipped", "step": step_name, "message": "Evidence sufficient — skipping web fallback", "index": i}
                continue

            # Standard step execution offloaded to thread to keep event loop responsive
            yield {"type": "step_progress", "step": step_name, "message": step_msg, "index": i}
            context = await asyncio.to_thread(step_fn, context)
            trace = context.pipeline_trace.steps[-1].model_dump() if context.pipeline_trace.steps else {}
            yield {"type": "step_done", "step": step_name, "trace": trace, "index": i}

