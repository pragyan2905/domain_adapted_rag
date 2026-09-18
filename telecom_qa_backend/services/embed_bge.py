from typing import List, Dict, Any, Tuple
try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    raise ImportError("Please install FlagEmbedding: pip install FlagEmbedding")

class BGEEmbedder:
    """
    Wrapper for the BAAI/bge-m3 model.
    Generates both dense semantic vectors and sparse lexical (BM25-style) vectors in a single pass.
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        # Setting use_fp16=True loads the model in float16, dramatically reducing VRAM usage
        # to ~1.2GB while maintaining near-perfect accuracy.
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def embed(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """
        Embeds a list of strings.
        
        Returns:
            Tuple containing:
            1. List of dense vectors (1024-dimensional float lists)
            2. List of sparse vectors (dicts mapping token_id strings to weights)
        """
        if not texts:
            return [], []
            
        # encode() returns a dict: {'dense_vecs': np.ndarray, 'lexical_weights': list[dict]}
        # We explicitly turn off colbert_vecs to save massive amounts of memory,
        # as we are only using Dense + Sparse (Hybrid) for retrieval, not late interaction.
        embeddings = self.model.encode(
            texts, 
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=False
        )
        
        dense_vecs = embeddings['dense_vecs'].tolist()
        
        # BGE-M3 returns lexical weights with integer token IDs. 
        # Qdrant requires string indices for sparse vectors in its Python client in some versions,
        # but modern qdrant-client handles integer keys. We will safely stringify them just in case.
        sparse_vecs = []
        for weight_dict in embeddings['lexical_weights']:
            sparse_vecs.append({str(k): float(v) for k, v in weight_dict.items()})
            
        return dense_vecs, sparse_vecs
