"""
evaluation/retrieval_eval.py
============================
Standalone Information Retrieval (IR) Benchmark for 3GPP RAG.

Evaluates the Retrieval & Ranking pipeline in isolation from the LLM generator:
  - Step 02: Representation (Dense + Sparse)
  - Step 03: Hybrid Vector Search
  - Step 04: Reciprocal Rank Fusion (RRF)
  - Step 06: BGE Cross-Encoder Reranking
  - Step 07: Evidence Selection

Standard IR Metrics Computed:
  - HitRate@1, HitRate@3, HitRate@5 (Recall at K)
  - MRR@5, MRR@10 (Mean Reciprocal Rank)
  - NDCG@5, NDCG@10 (Normalized Discounted Cumulative Gain)

Usage:
    cd telecom_qa_backend
    python -m evaluation.retrieval_eval
"""

import os
import sys
import math
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embed_bge import BGEEmbedder
from services.rerank_bge import BGEReranker
from services.qdrant_client import QdrantStore
from core.schemas import QueryContext
from pipeline import (
    step02_representation,
    step03_retrieval,
    step04_fusion,
    step05_filtering,
    step06_reranking,
    step07_selection
)

@dataclass
class RetrievalGoldPair:
    id: str
    query: str
    target_ts: str
    target_clause: str


RETRIEVAL_TEST_SET = [
    RetrievalGoldPair("ir-001", "What is the purpose of the RRC Setup procedure in 5G NR?", "38.331", "5.3.3"),
    RetrievalGoldPair("ir-002", "What is the RRCReconfiguration message used for in 5G NR?", "38.331", "5.3.5"),
    RetrievalGoldPair("ir-003", "RRC connection establishment timer T300 expiry actions", "38.331", "5.3.3.7"),
    RetrievalGoldPair("ir-004", "Role of AMF in 5G Core architecture and mobility", "23.501", "6.2.1"),
    RetrievalGoldPair("ir-005", "PDU session establishment and user plane handling", "23.501", "5.6"),
    RetrievalGoldPair("ir-006", "Network Repository Function NRF service registration discovery", "23.501", "6.2.6"),
    RetrievalGoldPair("ir-007", "Physical layer frequency ranges FR1 and FR2 operating bands", "38.101", "5.1"),
    RetrievalGoldPair("ir-008", "RRC connection re-establishment procedure initiation", "38.331", "5.3.7"),
]


def compute_dcg(relevances: List[int], k: int) -> float:
    """Computes Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg


def compute_ndcg(relevances: List[int], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank K."""
    dcg = compute_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = compute_dcg(ideal_relevances, k)
    return (dcg / idcg) if idcg > 0 else 0.0


def run_retrieval_benchmark():
    print("\n" + "=" * 65)
    print("  3GPP Isolated Retrieval & Ranking Benchmark")
    print("  Evaluating: Dense + Sparse + RRF + BGE Cross-Encoder")
    print("=" * 65 + "\n")

    embedder = BGEEmbedder(use_fp16=True)
    reranker = BGEReranker(use_fp16=True)
    qdrant = QdrantStore(host="localhost", port=6333)

    hit_at_1 = []
    hit_at_3 = []
    hit_at_5 = []
    reciprocal_ranks = []
    ndcg_at_5_list = []

    results = []

    for item in RETRIEVAL_TEST_SET:
        ctx = QueryContext(original_query=item.query)
        # Execute retrieval pipeline steps
        ctx = step02_representation.execute(ctx, embedder, enable_hyde=False)
        ctx = step03_retrieval.execute(ctx, qdrant)
        ctx = step04_fusion.execute(ctx)
        ctx = step05_filtering.execute(ctx)
        ctx = step06_reranking.execute(ctx, reranker)
        ctx = step07_selection.execute(ctx)

        # Check rankings
        candidates = ctx.selected_evidence
        rank_found = None
        relevances = []

        for rank, chunk in enumerate(candidates):
            meta = chunk.metadata
            is_match = (
                item.target_ts in meta.ts_number and
                (item.target_clause in meta.clause_id or meta.clause_id.startswith(item.target_clause))
            )
            relevances.append(1 if is_match else 0)
            if is_match and rank_found is None:
                rank_found = rank + 1

        rr = (1.0 / rank_found) if rank_found is not None else 0.0
        reciprocal_ranks.append(rr)
        hit_at_1.append(1 if rank_found == 1 else 0)
        hit_at_3.append(1 if rank_found is not None and rank_found <= 3 else 0)
        hit_at_5.append(1 if rank_found is not None and rank_found <= 5 else 0)
        ndcg_at_5_list.append(compute_ndcg(relevances, 5))

        status_tag = f"[RANK {rank_found}]" if rank_found else "[NOT FOUND]"
        print(f"  {item.id} | {item.query[:45]}... -> {status_tag}")

        results.append({
            "id": item.id,
            "query": item.query,
            "target_ts": item.target_ts,
            "target_clause": item.target_clause,
            "rank_found": rank_found,
            "reciprocal_rank": rr,
            "ndcg_at_5": round(ndcg_at_5_list[-1], 3)
        })

    mrr = round(statistics.mean(reciprocal_ranks), 3)
    hr1 = round(statistics.mean(hit_at_1), 3)
    hr3 = round(statistics.mean(hit_at_3), 3)
    hr5 = round(statistics.mean(hit_at_5), 3)
    mean_ndcg = round(statistics.mean(ndcg_at_5_list), 3)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_test_cases": len(RETRIEVAL_TEST_SET),
        "ir_metrics": {
            "mrr_mean_reciprocal_rank": mrr,
            "hit_rate_at_1": hr1,
            "hit_rate_at_3": hr3,
            "hit_rate_at_5": hr5,
            "ndcg_at_5": mean_ndcg,
        },
        "per_query_results": results
    }

    print("\n" + "=" * 65)
    print("  RETRIEVAL BENCHMARK SUMMARY (IR METRICS)")
    print("=" * 65)
    print(f"  MRR (Mean Reciprocal Rank) : {mrr * 100:.1f}%")
    print(f"  HitRate@1 (Top-1 Recall)   : {hr1 * 100:.1f}%")
    print(f"  HitRate@3 (Top-3 Recall)   : {hr3 * 100:.1f}%")
    print(f"  HitRate@5 (Top-5 Recall)   : {hr5 * 100:.1f}%")
    print(f"  NDCG@5 (Ranking Quality)   : {mean_ndcg * 100:.1f}%")
    print("=" * 65 + "\n")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"IR benchmark report saved to {out_file}\n")


if __name__ == "__main__":
    run_retrieval_benchmark()
