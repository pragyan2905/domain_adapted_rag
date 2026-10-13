import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.redis_cache import SemanticCache
from services.embed_bge import BGEEmbedder

def main():
    print("Loading embedder...")
    embedder = BGEEmbedder(use_fp16=True)
    
    print("Connecting to Redis...")
    redis_cache = SemanticCache(host="localhost", port=6380)
    
    # 50 QnA pairs related to telecom/3GPP
    qna_pairs = [
        {"q": f"What is the function of {x} in 5G?", "a": f"The function of {x} in 5G is to handle specific network control operations."} for x in [
            "AMF", "SMF", "UPF", "PCF", "UDM", "AUSF", "NSSF", "NEF", "NRF", "AF",
            "N3IWF", "UDR", "UDSF", "NWDAF", "CHF", "BSF", "SEPP", "NSSAAF", "EIR", "LMF",
            "GMLC", "SMLC", "CBC", "MME", "SGW", "PGW", "HSS", "PCRF", "eNodeB", "gNodeB"
        ]
    ]
    # Adding more to reach 50
    qna_pairs += [
        {"q": f"Explain 3GPP Release {x} features.", "a": f"3GPP Release {x} introduced enhancements for network efficiency and new capabilities."} for x in range(15, 35)
    ]
    
    print(f"Total QnA pairs: {len(qna_pairs)}")
    
    for i, pair in enumerate(qna_pairs):
        query = pair["q"]
        answer = pair["a"]
        
        # embed query
        query_dense, _ = embedder.embed([query])
        query_vector = query_dense[0]
        
        response_data = {
            "answer": answer,
            "citations": [],
            "is_abstained": False,
            "abstention_reason": None,
            "guardrails": {
                "jailbreak_detected": False,
                "pii_detected": False,
                "hallucination_detected": False,
                "answered_from_general_knowledge": False,
                "web_fallback_used": False,
                "is_out_of_domain": False
            }
        }
        
        redis_cache.save_to_cache(query_vector, json.dumps(response_data), ttl_seconds=86400 * 30)
        if (i+1) % 10 == 0:
            print(f"Inserted {i+1}/{len(qna_pairs)}")
            
    print("Done populating Redis!")

if __name__ == "__main__":
    main()
