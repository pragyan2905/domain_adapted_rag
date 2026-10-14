import os
import redis
import json
import uuid
import numpy as np
from typing import Optional, Dict, Any, List
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

class SemanticCache:
    """
    Wraps Redis Stack (RediSearch) to provide ultra-fast vector similarity caching.
    Supports both local Docker instances and cloud-managed Redis (via REDIS_URL).
    Gracefully falls back to key-value mode if RediSearch module is not present.
    """
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        redis_url: Optional[str] = None,
        index_name: str = "qa_cache_idx"
    ):
        url = redis_url or os.environ.get("REDIS_URL")
        if url:
            self.client = redis.from_url(url, decode_responses=False)
        else:
            self.client = redis.Redis(host=host, port=port, decode_responses=False)

        self.index_name = index_name
        self.vector_search_supported = True
        
        # Try to initialize RediSearch index, soft-failing if Redis or RediSearch is unavailable
        try:
            self._initialize_index()
        except Exception as e:
            self.vector_search_supported = False
            print(f"Redis Cache note: RediSearch index not active ({e}). Key-value cache and session memory remain functional.")

    def _initialize_index(self):
        try:
            self.client.ft(self.index_name).info()
        except redis.exceptions.ResponseError:
            # Index does not exist, create it.
            # 1024 dimensions for BGE-M3
            schema = (
                VectorField(
                    "vector",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": 1024, "DISTANCE_METRIC": "COSINE"}
                ),
                TextField("response_json")
            )
            definition = IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
            self.client.ft(self.index_name).create_index(fields=schema, definition=definition)
            print("Successfully created Redis Vector Index for Semantic Caching.")

    def check_cache(self, vector: List[float], threshold: float = 0.95) -> Optional[Dict[str, Any]]:
        """
        Performs a KNN search. If similarity > threshold, returns the cached JSON.
        """
        if not self.vector_search_supported:
            return None

        try:
            # RediSearch query syntax: find 1 nearest neighbor
            q = Query("*=>[KNN 1 @vector $vec AS score]").return_fields("response_json", "score").sort_by("score").dialect(2)
            
            vec_bytes = np.array(vector, dtype=np.float32).tobytes()
            res = self.client.ft(self.index_name).search(q, {"vec": vec_bytes})
            
            if res.docs:
                doc = res.docs[0]
                # In Redis COSINE metric, 'score' is actually the distance.
                # similarity = 1 - distance
                distance = float(doc.score)
                similarity = 1.0 - distance
                
                if similarity >= threshold:
                    print(f"SEMANTIC CACHE HIT! Similarity: {similarity:.4f}")
                    # RediSearch automatically decodes strings, so it's already a str
                    return json.loads(doc.response_json)
            return None
        except Exception as e:
            print(f"Redis Cache Check failed (soft fail): {e}")
            return None

    def save_to_cache(self, vector: List[float], response_json: str, ttl_seconds: int):
        """
        Saves the generated answer to Redis with a TTL.
        """
        if ttl_seconds <= 0 or not self.vector_search_supported:
            return
            
        try:
            query_id = str(uuid.uuid4())
            key = f"cache:{query_id}"
            
            vec_bytes = np.array(vector, dtype=np.float32).tobytes()
            mapping = {
                b"vector": vec_bytes,
                b"response_json": response_json.encode('utf-8')
            }
            
            self.client.hset(key, mapping=mapping)
            self.client.expire(key, ttl_seconds)
            print(f"Saved to Semantic Cache. TTL: {ttl_seconds}s")
        except Exception as e:
            print(f"Failed to save to Redis Cache: {e}")
