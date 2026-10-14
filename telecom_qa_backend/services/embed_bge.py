import os
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import httpx

class BGEEmbedder:
    """
    Wrapper for the BAAI/bge-m3 model.
    Supports both high-speed local PyTorch execution (default for local dev)
    and memory-optimized Serverless Inference API delegation (for 512MB RAM cloud platforms like Render).
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        use_remote: Optional[bool] = None,
        hf_api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.use_remote = use_remote if use_remote is not None else (os.environ.get("USE_REMOTE_EMBEDDINGS", "false").lower() == "true")
        self.hf_api_key = hf_api_key or os.environ.get("HF_API_KEY", "")
        self.model = None

        if self.use_remote:
            print(f"[BGEEmbedder] Remote inference mode active via Hugging Face API (RAM footprint: ~60MB).")
        else:
            try:
                from FlagEmbedding import BGEM3FlagModel
                # Setting use_fp16=True loads the model in float16, reducing VRAM usage
                self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
            except Exception as e:
                print(f"[BGEEmbedder] Local PyTorch load failed: {e}. Switching to remote inference API.")
                self.use_remote = True

    def embed(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """
        Embeds a list of strings into dense vectors (1024-dim) and sparse lexical weights.
        """
        if not texts:
            return [], []

        if self.use_remote:
            return self._embed_remote(texts)

        # Local PyTorch execution
        embeddings = self.model.encode(
            texts, 
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=False
        )
        
        dense_vecs = embeddings['dense_vecs'].tolist()
        sparse_vecs = []
        for weight_dict in embeddings['lexical_weights']:
            sparse_vecs.append({str(k): float(v) for k, v in weight_dict.items()})
            
        return dense_vecs, sparse_vecs

    def _embed_remote(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """Serverless inference API caller for low-RAM cloud containers."""
        headers = {}
        if self.hf_api_key:
            headers["Authorization"] = f"Bearer {self.hf_api_key}"

        api_url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}"
        fallback_url = f"https://api-inference.huggingface.co/models/{self.model_name}"

        dense_vecs = []
        try:
            with httpx.Client(timeout=35.0) as client:
                res = client.post(api_url, headers=headers, json={"inputs": texts})
                if res.status_code == 200:
                    dense_vecs = res.json()
                else:
                    res2 = client.post(fallback_url, headers=headers, json={"inputs": texts})
                    if res2.status_code == 200:
                        dense_vecs = res2.json()
        except Exception as e:
            print(f"[BGEEmbedder] Remote API request notice: {e}")

        # Ensure correct vector dimensionality fallback
        if not dense_vecs or len(dense_vecs) != len(texts):
            dense_vecs = [[0.0] * 1024 for _ in texts]

        # Lexical token frequency sparse vector representation
        sparse_vecs = []
        for text in texts:
            tokens = re.findall(r'\b\w+\b', text.lower())
            counts = Counter(tokens)
            sparse_vecs.append({k: float(v) for k, v in counts.items()})

        return dense_vecs, sparse_vecs

