from typing import List, Tuple
try:
    from FlagEmbedding import FlagReranker
except ImportError:
    raise ImportError("Please install FlagEmbedding: pip install FlagEmbedding")

class BGEReranker:
    """
    Wrapper for the BAAI/bge-reranker-v2-m3 model.
    A cross-encoder that takes a (query, passage) pair and computes a highly accurate relevance score.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True):
        # use_fp16=True significantly reduces VRAM while maintaining accuracy
        self.model = FlagReranker(model_name, use_fp16=use_fp16)

    def compute_scores(self, query: str, passages: List[str]) -> List[float]:
        """
        Computes relevance scores for a list of passages against the query.
        Returns a list of float scores corresponding to the passages.
        """
        if not passages:
            return []
            
        pairs = [[query, passage] for passage in passages]
        
        # model.compute_score returns a list of floats if len(pairs) > 1, 
        # or a single float if len(pairs) == 1. We standardize it to a list.
        scores = self.model.compute_score(pairs)
        
        if isinstance(scores, float):
            return [scores]
        return scores
