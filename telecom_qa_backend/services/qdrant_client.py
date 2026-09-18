import uuid
from typing import List, Dict, Any, Tuple
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        VectorParams, Distance, PointStruct, 
        SparseVectorParams, SparseVector,
        Filter, FieldCondition, MatchValue
    )
except ImportError:
    raise ImportError("Please install qdrant-client: pip install qdrant-client")

from core.schemas import ChunkMetadata, RetrievedChunk

class QdrantStore:
    """
    Interface to the self-hosted Qdrant Docker container.
    Handles storage and retrieval of Dense and Sparse vectors.
    """
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "3gpp_standards"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates the collection with both Dense and Sparse vector support if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=1024, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                }
            )

    def insert_chunks(self, chunks: List[Dict[str, Any]], dense_vecs: List[List[float]], sparse_vecs: List[Dict[str, float]]):
        """
        Inserts chunks into Qdrant. 
        `chunks` is the list of dicts returned by ClauseAwareChunker.
        """
        points = []
        for i, chunk in enumerate(chunks):
            # Parse sparse dict into Qdrant SparseVector format
            # Sparse vectors are dicts of {str_token_id: float_weight}
            indices = [int(k) for k in sparse_vecs[i].keys()]
            values = list(sparse_vecs[i].values())
            
            point = PointStruct(
                id=chunk["metadata"].chunk_id,
                payload={
                    "text": chunk["text"],
                    **chunk["metadata"].model_dump()
                },
                vector={
                    "dense": dense_vecs[i],
                    "sparse": SparseVector(indices=indices, values=values)
                }
            )
            points.append(point)
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search_dense(self, dense_vec: List[float], limit: int = 15, ts_filters: List[str] = None) -> List[RetrievedChunk]:
        """Searches using only the dense semantic vector."""
        query_filter = self._build_filter(ts_filters)
        
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vec,
            using="dense",
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )
        return self._parse_results(response.points)

    def search_sparse(self, sparse_vec: Dict[str, float], limit: int = 15, ts_filters: List[str] = None) -> List[RetrievedChunk]:
        """Searches using only the sparse lexical vector (BM25 equivalent)."""
        query_filter = self._build_filter(ts_filters)
        
        indices = [int(k) for k in sparse_vec.keys()]
        values = list(sparse_vec.values())
        
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(indices=indices, values=values),
            using="sparse",
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )
        return self._parse_results(response.points)

    def _build_filter(self, ts_filters: List[str] = None) -> Filter | None:
        """Helper to build Qdrant metadata filters."""
        if not ts_filters:
            return None
            
        # Example: if query detected we only want "38.331", filter by it
        conditions = []
        for ts in ts_filters:
            conditions.append(
                FieldCondition(key="ts_number", match=MatchValue(value=ts))
            )
            
        return Filter(should=conditions)

    def _parse_results(self, raw_results) -> List[RetrievedChunk]:
        """Converts Qdrant payload back into our strict Pydantic schemas."""
        parsed = []
        for res in raw_results:
            payload = res.payload
            text = payload.pop("text")
            
            # The rest of the payload should match ChunkMetadata
            metadata = ChunkMetadata(**payload)
            
            parsed.append(RetrievedChunk(
                text=text,
                metadata=metadata,
                score=res.score
            ))
        return parsed
