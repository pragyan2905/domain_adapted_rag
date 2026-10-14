"""
evaluation/concurrency_test.py
==============================
Stress and load testing harness for 3GPP Telecom RAG.
Evaluates concurrency, throughput (RPS), and latency distribution (P50/P95/P99)
across both local and cloud-deployed environments.

Usage:
    # Test local backend with 5 concurrent requests:
    python -m evaluation.concurrency_test --url http://127.0.0.1:8001 --concurrency 5

    # Test deployed cloud instance:
    python -m evaluation.concurrency_test --url https://your-deployed-app.com --concurrency 3

    # Test concurrent semantic cache performance:
    python -m evaluation.concurrency_test --concurrency 10 --cached
"""

import os
import sys
import time
import json
import asyncio
import argparse
import statistics
from datetime import datetime
from typing import List, Dict, Any

try:
    import httpx
except ImportError:
    print("Please install httpx: pip install httpx")
    sys.exit(1)

BENCHMARK_QUERIES = [
    "What is the purpose of the RRC Setup procedure in 5G NR?",
    "Explain the concept of a PDU session in 5G.",
    "What is the role of the AMF in 5G Core?",
    "Describe the RRC states in 5G NR and transitions between them.",
    "What is the NRF and how does service discovery work in 5GC?",
    "What are the frequency ranges FR1 and FR2 in 5G NR?",
    "What is the RRCReconfiguration message used for in 5G NR Rel-17?",
    "How does network slicing work at the core network level in 5G?",
]


async def send_query(
    client: httpx.AsyncClient,
    base_url: str,
    query: str,
    session_id: str,
    request_id: int
) -> Dict[str, Any]:
    """Sends a single query and measures exact turnaround latency."""
    url = f"{base_url.rstrip('/')}/query"
    payload = {"query": query, "session_id": session_id}
    start = time.perf_counter()
    
    try:
        response = await client.post(url, json=payload, timeout=90.0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        status = response.status_code
        data = response.json() if status == 200 else {}
        is_cache_hit = False
        if status == 200 and "pipeline_trace" in data:
            is_cache_hit = data["pipeline_trace"].get("cache_hit", False)

        return {
            "request_id": request_id,
            "query": query,
            "status_code": status,
            "latency_ms": elapsed_ms,
            "cache_hit": is_cache_hit,
            "is_abstained": data.get("is_abstained", False),
            "citations_count": len(data.get("citations", [])),
            "error": None if status == 200 else response.text[:200]
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "request_id": request_id,
            "query": query,
            "status_code": 0,
            "latency_ms": elapsed_ms,
            "cache_hit": False,
            "is_abstained": False,
            "citations_count": 0,
            "error": str(e)
        }


async def run_load_test(
    base_url: str,
    concurrency: int,
    total_requests: int,
    use_cached_query: bool = False
) -> Dict[str, Any]:
    """Executes concurrent requests with bounded semaphore concurrency."""
    print("\n" + "=" * 65)
    print(f"  3GPP RAG Concurrency Benchmark")
    print(f"  Target URL      : {base_url}")
    print(f"  Concurrency     : {concurrency} parallel workers")
    print(f"  Total Requests  : {total_requests}")
    print(f"  Mode            : {'Cached Query Intercept' if use_cached_query else 'Cold Diverse Queries'}")
    print("=" * 65 + "\n")

    # If cached mode, warmup once so the query is in Redis semantic cache
    cached_warmup_query = "What is the purpose of the RRC Setup procedure in 5G NR?"
    if use_cached_query:
        print("[Warmup] Sending initial request to warm semantic cache...")
        async with httpx.AsyncClient() as client:
            warmup_res = await client.post(
                f"{base_url.rstrip('/')}/query",
                json={"query": cached_warmup_query, "session_id": "warmup-session"},
                timeout=60.0
            )
            print(f"[Warmup] Status: {warmup_res.status_code} (Cache primed)")

    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(limits=limits) as client:
        async def worker(req_id: int):
            async with semaphore:
                query = cached_warmup_query if use_cached_query else BENCHMARK_QUERIES[req_id % len(BENCHMARK_QUERIES)]
                session_id = f"concurrency-session-{req_id % concurrency}"
                print(f"[{req_id+1:02d}/{total_requests:02d}] Sent: '{query[:40]}...'")
                res = await send_query(client, base_url, query, session_id, req_id + 1)
                cache_tag = "[CACHE]" if res["cache_hit"] else "[COLD]"
                print(f"[{req_id+1:02d}/{total_requests:02d}] Done: HTTP {res['status_code']} | {cache_tag} | {res['latency_ms']:.1f}ms")
                return res

        start_time = time.perf_counter()
        tasks = [worker(i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks)
        total_time_sec = time.perf_counter() - start_time

    # Calculate statistics
    latencies = [r["latency_ms"] for r in results if r["status_code"] == 200]
    all_latencies = [r["latency_ms"] for r in results]
    successes = sum(1 for r in results if r["status_code"] == 200)
    rate_limited = sum(1 for r in results if r["status_code"] == 429)
    errors = sum(1 for r in results if r["status_code"] not in [200, 429])
    cache_hits = sum(1 for r in results if r["cache_hit"])

    lat_p50 = statistics.median(latencies) if latencies else 0.0
    lat_p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else 0.0)
    lat_min = min(latencies) if latencies else 0.0
    lat_max = max(latencies) if latencies else 0.0
    lat_avg = statistics.mean(latencies) if latencies else 0.0
    rps = total_requests / total_time_sec if total_time_sec > 0 else 0.0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "target_url": base_url,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "total_time_sec": round(total_time_sec, 2),
        "requests_per_sec": round(rps, 2),
        "success_count": successes,
        "rate_limited_count": rate_limited,
        "error_count": errors,
        "cache_hits": cache_hits,
        "latency_stats_ms": {
            "min": round(lat_min, 1),
            "avg": round(lat_avg, 1),
            "p50_median": round(lat_p50, 1),
            "p95": round(lat_p95, 1),
            "max": round(lat_max, 1),
        },
        "individual_results": results
    }

    # Print Summary Table
    print("\n" + "=" * 65)
    print("  CONCURRENCY BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"  Total Duration       : {total_time_sec:.2f} s")
    print(f"  Throughput (RPS)     : {rps:.2f} req/sec")
    print(f"  Success Rate         : {successes}/{total_requests} ({(successes/total_requests)*100:.1f}%)")
    print(f"  Rate Limited (429)   : {rate_limited}")
    print(f"  Errors               : {errors}")
    print(f"  Cache Hits           : {cache_hits}")
    print("  -------------------------------------------------------------")
    print(f"  Latency Min          : {lat_min:.1f} ms")
    print(f"  Latency Average      : {lat_avg:.1f} ms")
    print(f"  Latency P50 (Median) : {lat_p50:.1f} ms")
    print(f"  Latency P95          : {lat_p95:.1f} ms")
    print(f"  Latency Max          : {lat_max:.1f} ms")
    print("=" * 65 + "\n")

    # Save to results directory
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"concurrency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Detailed logs saved to {out_file}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="3GPP Telecom RAG Concurrency & Stress Tester")
    parser.add_argument("--url", default=os.environ.get("QA_API_URL", "http://127.0.0.1:8001"), help="Target API URL")
    parser.add_argument("--concurrency", "-c", type=int, default=3, help="Concurrent client requests (default: 3)")
    parser.add_argument("--total", "-n", type=int, default=6, help="Total requests to dispatch (default: 6)")
    parser.add_argument("--cached", action="store_true", help="Send identical query to evaluate semantic cache concurrency")

    args = parser.parse_args()
    asyncio.run(run_load_test(
        base_url=args.url,
        concurrency=args.concurrency,
        total_requests=args.total,
        use_cached_query=args.cached
    ))


if __name__ == "__main__":
    main()
