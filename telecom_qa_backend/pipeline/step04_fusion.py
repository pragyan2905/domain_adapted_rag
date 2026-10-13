from typing import Dict
from core.schemas import QueryContext, RetrievedChunk
from core.tracer import trace_step

def execute(context: QueryContext, k: int = 60) -> QueryContext:
    """
    Step 4: Candidate Fusion
    Implements Reciprocal Rank Fusion (RRF) to merge the results of the
    Dense and Sparse retrieval steps into a single, ranked candidate list.

    Formula: RRF_Score = sum(1 / (k + rank))
    """
    total_input = len(context.dense_candidates) + len(context.sparse_candidates)

    with trace_step(context, "Step 04: RRF Fusion", input_count=total_input) as t:
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        # 1. Rank Dense Candidates
        for rank, chunk in enumerate(context.dense_candidates):
            cid = chunk.metadata.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = chunk.model_copy(deep=True)
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank + 1))

        # 2. Rank Sparse Candidates
        for rank, chunk in enumerate(context.sparse_candidates):
            cid = chunk.metadata.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = chunk.model_copy(deep=True)
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank + 1))

        # 3. Sort by RRF score descending
        sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # 4. Assign the new fusion score and store in context
        for cid in sorted_cids:
            chunk = chunk_map[cid]
            chunk.score = scores[cid]
            context.fused_candidates.append(chunk)

        t["output_count"] = len(context.fused_candidates)
        t["extra"] = {
            "unique_chunks_after_fusion": len(context.fused_candidates),
            "top_rrf_score": round(context.fused_candidates[0].score, 5) if context.fused_candidates else 0,
        }

    return context
