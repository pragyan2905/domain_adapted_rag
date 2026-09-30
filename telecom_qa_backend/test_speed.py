import requests
import time
import json

url = "http://127.0.0.1:8001/query"
payload = {"query": "what is 5g technology ?"}

print("--- ATTEMPT 1 (Forcing Cache Miss by slightly modifying string if needed, but assuming empty cache for this exact string) ---")
start = time.time()
res1 = requests.post(url, json=payload)
end = time.time()
data1 = res1.json()

print(f"Time Taken: {end - start:.4f} seconds")
print(f"Answer Quality (first 300 chars):\n{data1.get('answer')[:300]}...\n")
print(f"Guardrails: {json.dumps(data1.get('guardrails'), indent=2)}\n")

print("--- ATTEMPT 2 (Expected Cache Hit) ---")
start2 = time.time()
res2 = requests.post(url, json=payload)
end2 = time.time()
data2 = res2.json()

print(f"Time Taken: {end2 - start2:.4f} seconds")
print(f"Answer Quality (first 300 chars):\n{data2.get('answer')[:300]}...\n")
