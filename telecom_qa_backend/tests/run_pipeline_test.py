import os
import sys

# Ensure the root of the project is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schemas import QueryContext
from services.qdrant_client import QdrantStore
from services.embed_bge import BGEEmbedder
from services.rerank_bge import BGEReranker
from services.llm_gemini import GeminiGenerator
from services.search_duckduckgo import DuckDuckGoSearch

from pipeline import (
    step01_understanding, step02_representation, step03_retrieval, step04_fusion,
    step05_filtering, step06_reranking, step07_selection, step08_validation,
    step09_web_fallback, step10_abstention, step11_generation, step12_guardrails
)

def run_test():
    from dotenv import load_dotenv
    load_dotenv()
    
    API_KEY = os.environ.get("GEMINI_API_KEY")
    if not API_KEY:
        print("ERROR: Please set GEMINI_API_KEY environment variable before running.")
        print("Example: export GEMINI_API_KEY='your_key'")
        return

    print("Initializing ML Models and Services. This will take a moment to load into VRAM...")
    
    try:
        embedder = BGEEmbedder(use_fp16=True)
        reranker = BGEReranker(use_fp16=True)
        qdrant = QdrantStore(host="localhost", port=6333)
        web_search = DuckDuckGoSearch()
        generator = GeminiGenerator(api_key=API_KEY)
    except Exception as e:
        print(f"\nFailed to initialize services: {e}")
        print("Ensure you have run `docker compose up -d` and installed requirements.")
        return
        
    test_query = "What is the RRC connection setup timer in Rel-17?"
    print(f"\n--- Running Pipeline for Query: '{test_query}' ---\n")
    
    ctx = QueryContext(original_query=test_query)
    
    steps = [
        ("Step  1: Understanding", step01_understanding.execute, [generator]),
        ("Step  2: Representation", step02_representation.execute, [embedder]),
        ("Step  3: Retrieval", step03_retrieval.execute, [qdrant]),
        ("Step  4: Fusion", step04_fusion.execute, []),
        ("Step  5: Filtering", step05_filtering.execute, []),
        ("Step  6: Reranking", step06_reranking.execute, [reranker]),
        ("Step  7: Selection", step07_selection.execute, []),
        ("Step  8: Validation", step08_validation.execute, [generator]),
        ("Step  9: Web Fallback", step09_web_fallback.execute, [web_search, generator]),
        ("Step 10: Abstention", step10_abstention.execute, []),
        ("Step 11: Generation", step11_generation.execute, [generator]),
        ("Step 12: Guardrails", step12_guardrails.execute, [])
    ]
    
    for name, func, args in steps:
        print(f"Executing {name}...")
        ctx = func(ctx, *args)
        
    print("\n--- Pipeline Complete! ---\n")
    print("FINAL ANSWER:")
    print(ctx.final_answer)
    print("\nCITATIONS:", ctx.citations)
    print("ABSTAINED:", ctx.is_abstained)
    
    print("\n--- Full Context Dump (The Master Folder) ---")
    # This will print the massive JSON log so you can inspect every intermediate step
    print(ctx.model_dump_json(indent=2))

if __name__ == "__main__":
    run_test()
