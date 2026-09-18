from typing import Dict
from core.schemas import QueryContext, RetrievedChunk

def execute(context: QueryContext, k: int = 60) -> QueryContext:
    """
    Step 4: Candidate Fusion
    Implements Reciprocal Rank Fusion (RRF) to merge the results of the 
    Dense and Sparse retrieval steps into a single, ranked candidate list.
    
    Formula: RRF_Score = sum(1 / (k + rank))
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, RetrievedChunk] = {}
    
    # 1. Rank Dense Candidates
    for rank, chunk in enumerate(context.dense_candidates):
        cid = chunk.metadata.chunk_id
        if cid not in chunk_map:
            # Create a copy so we don't mutate the original retrieved chunk's score
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
        chunk.score = scores[cid]  # Overwrite with fusion score
        context.fused_candidates.append(chunk)
        
    return context
