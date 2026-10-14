import os
import re
from typing import List, Tuple, Optional
import httpx

class BGEReranker:
    """
    Wrapper for the BAAI/bge-reranker-v2-m3 model.
    A cross-encoder that takes a (query, passage) pair and computes a highly accurate relevance score.
    Supports local PyTorch inference or cloud inference API delegation for low-RAM hosts.
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        use_remote: Optional[bool] = None,
        hf_api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.use_remote = use_remote if use_remote is not None else (os.environ.get("USE_REMOTE_EMBEDDINGS", "false").lower() == "true")
        self.hf_api_key = hf_api_key or os.environ.get("HF_API_KEY", "")
        self.model = None

        if self.use_remote:
            print(f"[BGEReranker] Remote cross-encoder inference active (RAM footprint: ~20MB).")
        else:
            try:
                from FlagEmbedding import FlagReranker
                self.model = FlagReranker(model_name, use_fp16=use_fp16)
            except Exception as e:
                print(f"[BGEReranker] Local reranker load notice: {e}. Switching to remote mode.")
                self.use_remote = True

    def compute_scores(self, query: str, passages: List[str]) -> List[float]:
        """
        Computes relevance scores for a list of passages against the query.
        Returns a list of float scores corresponding to the passages.
        """
        if not passages:
            return []

        if self.use_remote:
            return self._compute_scores_remote(query, passages)

        pairs = [[query, passage] for passage in passages]
        scores = self.model.compute_score(pairs)
        
        if isinstance(scores, float):
            return [scores]
        return scores

    def _compute_scores_remote(self, query: str, passages: List[str]) -> List[float]:
        """Calculates reranker cross-encoder scores via inference API or term-density heuristic."""
        headers = {}
        if self.hf_api_key:
            headers["Authorization"] = f"Bearer {self.hf_api_key}"

        api_url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}"
        pairs = [[query, passage] for passage in passages]

        try:
            with httpx.Client(timeout=35.0) as client:
                res = client.post(api_url, headers=headers, json={"inputs": pairs})
                if res.status_code == 200:
                    scores = res.json()
                    if isinstance(scores, list):
                        return [float(s if not isinstance(s, dict) else s.get("score", 0.0)) for s in scores]
        except Exception as e:
            print(f"[BGEReranker] Remote rerank request note: {e}")

        # Term-overlap ranking heuristic fallback if remote API is unauthenticated or busy
        q_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        heuristic_scores = []
        for passage in passages:
            p_tokens = set(re.findall(r'\b\w+\b', passage.lower()))
            overlap = len(q_tokens.intersection(p_tokens))
            heuristic_scores.append(float(overlap))
        return heuristic_scores

