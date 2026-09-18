import os
import sys

# Ensure the root of the project is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chunking import ClauseAwareChunker
from services.embed_bge import BGEEmbedder
from services.qdrant_client import QdrantStore

def ingest_file(filepath: str, ts_number: str, series: str, release_version: str):
    """
    Reads a raw text file, chunks it via 3GPP rules, embeds it, and pushes it to Qdrant.
    """
    if not os.path.exists(filepath):
        print(f"ERROR: File not found -> {filepath}")
        return

    print(f"\n--- Starting Ingestion for {ts_number} ({release_version}) ---")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        
    print(f"Read {len(raw_text)} characters from {filepath}.")

    # 1. Chunking
    print("Chunking document using 3GPP clause-aware rules...")
    chunker = ClauseAwareChunker()
    chunks = chunker.chunk_document(
        text=raw_text, 
        ts_number=ts_number, 
        series=series, 
        release_version=release_version
    )
    print(f"Generated {len(chunks)} strict chunks.")

    # 2. Embedding
    print("Loading BGE-M3 Embedder (This will take VRAM)...")
    embedder = BGEEmbedder(use_fp16=True)
    
    print("Generating Dense and Sparse embeddings...")
    # Extract just the text payload for the embedder
    texts_to_embed = [c["text"] for c in chunks]
    dense_vecs, sparse_vecs = embedder.embed(texts_to_embed)

    # 3. Database Insertion
    print("Connecting to Qdrant...")
    try:
        qdrant = QdrantStore()
        qdrant.insert_chunks(chunks, dense_vecs, sparse_vecs)
        print("Successfully inserted all chunks into Qdrant!")
    except Exception as e:
        print(f"Failed to insert into Qdrant. Is Docker running? Error: {e}")

if __name__ == "__main__":
    import glob
    import re
    
    # Process all text files in the data directory
    data_files = glob.glob("data/ts_*_rel17.txt")
    
    if not data_files:
        print("No files found in data/ directory. Run fetch_3gpp.py first.")
        sys.exit(0)
        
    for filepath in data_files:
        # Extract the TS number from the filename, e.g. "ts_38_331_rel17.txt" -> "38.331"
        basename = os.path.basename(filepath)
        match = re.search(r'ts_(\d+)_(\d+)_rel17', basename)
        if match:
            ts_number = f"{match.group(1)}.{match.group(2)}"
            ingest_file(
                filepath=filepath,
                ts_number=ts_number,
                series="38-series",
                release_version="Rel-17"
            )
        else:
            print(f"Skipping {filepath} - couldn't parse TS number from filename.")
