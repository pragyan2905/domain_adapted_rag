import requests
import time
import json
import os

API_URL = "http://127.0.0.1:8001/query"

# A highly curated list of foundational 5G concepts.
# By caching these, we create a mathematical "net" that will intercept 
# hundreds of differently phrased variations of these questions.
CORE_QUESTIONS = [
    "What is 5G NR?",
    "What is the difference between 5G Standalone (SA) and Non-Standalone (NSA)?",
    "What is a gNB in 5G architecture?",
    "Explain the concept of Network Slicing in 5G.",
    "What are the primary frequency bands used in 5G (FR1 and FR2)?",
    "What is Massive MIMO and how does it work in 5G?",
    "Describe the 5G Core (5GC) architecture and its main network functions.",
    "What is the role of the AMF (Access and Mobility Management Function)?",
    "What is the role of the SMF (Session Management Function)?",
    "What is the role of the UPF (User Plane Function)?",
    "How does 5G handle Quality of Service (QoS)?",
    "What is the 5G QoS Identifier (5QI)?",
    "Explain the concepts of eMBB, URLLC, and mMTC in 5G.",
    "What is Beamforming and why is it essential for 5G mmWave?",
    "What is the difference between an EPC and a 5GC?",
    "How does 5G reduce latency compared to 4G LTE?",
    "What is a PDU Session in 5G?",
    "Explain the 5G security architecture and Authentication and Key Agreement (AKA).",
    "What is Multi-Access Edge Computing (MEC) in the context of 5G?",
    "What is the purpose of the NRF (Network Repository Function)?"
]

def warm_cache():
    print(f"Starting Semantic Cache Warming sequence for {len(CORE_QUESTIONS)} core concepts...")
    print("This will take some time. Delaying between requests to respect Gemini API rate limits.\n")
    
    successful = 0
    failed = 0
    
    for i, question in enumerate(CORE_QUESTIONS, 1):
        print(f"[{i}/{len(CORE_QUESTIONS)}] Pinging Backend: '{question}'")
        
        try:
            # We set a long timeout (120s) because the backend itself has a 3-retry 
            # exponential backoff if Gemini throws a 503 High Demand error.
            start_time = time.time()
            response = requests.post(API_URL, json={"query": question}, timeout=120)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                print(f"   -> Success! (Took {elapsed:.2f}s). Cached in Redis.")
                successful += 1
            else:
                print(f"   -> Failed with status {response.status_code}: {response.text}")
                failed += 1
                
        except requests.exceptions.RequestException as e:
            print(f"   -> Request Exception: {e}")
            failed += 1
            
        # VERY IMPORTANT: To avoid being permanently IP banned or rate-limited by Gemini 
        # (especially during their current global 503 outage), we force a 15-second cool-down.
        if i < len(CORE_QUESTIONS):
            print("   -> Sleeping for 15 seconds to let Gemini API breathe...")
            time.sleep(15)
            
    print("\n" + "="*50)
    print(f"Cache Warming Complete!")
    print(f"Successfully cached: {successful}/{len(CORE_QUESTIONS)}")
    print(f"Failed: {failed}/{len(CORE_QUESTIONS)}")
    print("="*50)

if __name__ == "__main__":
    # Ensure backend is actually reachable before starting
    try:
        requests.options(API_URL, timeout=5)
        warm_cache()
    except requests.exceptions.ConnectionError:
        print(f"CRITICAL ERROR: Could not connect to {API_URL}.")
        print("Make sure you have started your backend terminal (python -m uvicorn api.main:app --host 0.0.0.0 --port 8001)")
