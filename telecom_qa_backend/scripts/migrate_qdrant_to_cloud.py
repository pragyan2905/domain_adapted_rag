import os
import sys
import time
from dotenv import load_dotenv

# Load env variables
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, 
    SparseVectorParams, SparseVector
)

def migrate():
    cloud_url = os.environ.get("QDRANT_URL")
    cloud_api_key = os.environ.get("QDRANT_API_KEY")

    if not cloud_url or not cloud_api_key:
        print("ERROR: QDRANT_URL and QDRANT_API_KEY must be set in .env")
        sys.exit(1)

    print(f"[Migration] Connecting to Local Qdrant (localhost:6333)...")
    local_client = QdrantClient(host="localhost", port=6333)
    
    print(f"[Migration] Connecting to Cloud Qdrant ({cloud_url})...")
    cloud_client = QdrantClient(url=cloud_url, api_key=cloud_api_key)

    collection_name = "3gpp_standards"

    # Check local collection
    if not local_client.collection_exists(collection_name):
        print(f"ERROR: Local collection '{collection_name}' does not exist!")
        sys.exit(1)

    local_count = local_client.get_collection(collection_name).points_count
    print(f"[Migration] Found {local_count} points in local '{collection_name}'.")

    # Ensure cloud collection exists with dual dense + sparse configuration
    if not cloud_client.collection_exists(collection_name):
        print(f"[Migration] Creating collection '{collection_name}' in Qdrant Cloud...")
        cloud_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            }
        )
    else:
        print(f"[Migration] Collection '{collection_name}' already exists in Qdrant Cloud.")

    print("[Migration] Scrolling points from local Qdrant and uploading to Cloud in batches of 100...")
    batch_size = 100
    migrated_count = 0
    next_offset = None
    start_time = time.perf_counter()

    while True:
        records, next_offset = local_client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=next_offset,
            with_payload=True,
            with_vectors=True
        )

        if not records:
            break

        points_to_upsert = []
        for r in records:
            points_to_upsert.append(PointStruct(
                id=r.id,
                vector=r.vector,
                payload=r.payload
            ))

        cloud_client.upsert(
            collection_name=collection_name,
            points=points_to_upsert
        )
        migrated_count += len(records)
        print(f"  -> Uploaded {migrated_count}/{local_count} points ({migrated_count/local_count*100:.1f}%)...")

        if next_offset is None:
            break

    duration = time.perf_counter() - start_time
    cloud_final_count = cloud_client.get_collection(collection_name).points_count
    print(f"\n[Migration Complete] Successfully migrated {migrated_count} points in {duration:.2f}s!")
    print(f"[Verification] Qdrant Cloud points count: {cloud_final_count}")

if __name__ == "__main__":
    migrate()
