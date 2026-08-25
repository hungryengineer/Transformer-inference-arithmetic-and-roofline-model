import statistics
import sys
import time
import requests
import random
import concurrent.futures

ENDPOINTS = {
    "rr": "http://localhost:8000/v1/chat/completions",
    "prefix": "http://192.168.1.40:8000/v1/chat/completions",
}

# Create 8 distinct large prefixes
PREFIXES = []
for p_idx in range(8):
    prefix = " ".join(
        f"Shared incident context {p_idx} section {i}: "
        "production Kubernetes inference systems require careful "
        "capacity planning, observability, scheduling, and recovery."
        for i in range(1, 70)
    )
    PREFIXES.append(prefix)

def make_request(url, request_id):
    # Cycle through the prefixes to ensure a diverse mix
    p_idx = request_id % len(PREFIXES)
    prefix = PREFIXES[p_idx]
    
    payload = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [{
            "role": "user",
            "content": (
                prefix
                + f"\n\nUnique request {request_id}: analyze this incident "
                  "and summarize the likely cause."
            ),
        }],
        "max_tokens": 40,
        "temperature": 0,
    }

    start = time.perf_counter()
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200:
        print(f"Error {r.status_code}: {r.text}")
    r.raise_for_status()
    r.json()
    return time.perf_counter() - start

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <target>")
        print(f"Available targets: {', '.join(ENDPOINTS.keys())}")
        sys.exit(1)

    target = sys.argv[1]
    if target not in ENDPOINTS:
        print(f"Invalid target: {target}. Available targets: {', '.join(ENDPOINTS.keys())}")
        sys.exit(1)
        
    url = ENDPOINTS[target]

    latencies = []
    
    print(f"Starting test for target: {target} with 80 concurrent requests across 8 distinct prefixes...")
    
    # Use ThreadPoolExecutor to send concurrent requests and simulate real load
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, url, i) for i in range(80)]
        for future in concurrent.futures.as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception as e:
                print(f"Request failed: {e}")

    if not latencies:
        print("No successful requests")
        return

    p95 = statistics.quantiles(latencies, n=20)[18]
    p50 = statistics.median(latencies)

    print(f"\n--- RESULTS FOR {target.upper()} ---")
    print(f"target={target}")
    print(f"requests={len(latencies)}")
    print(f"p95_latency={p95:.4f}s")
    print(f"p50_latency={p50:.4f}s")
    print(f"min={min(latencies):.4f}s")
    print(f"max={max(latencies):.4f}s")
    print("-----------------------------\n")

if __name__ == "__main__":
    main()
