import statistics
import sys
import time
import requests

ENDPOINTS = {
    "rr": "http://localhost:8000/v1/chat/completions",
    "prefix": "http://192.168.1.40:8000/v1/chat/completions",
}

PREFIX = " ".join(
    f"Shared incident context section {i}: "
    "production Kubernetes inference systems require careful "
    "capacity planning, observability, scheduling, and recovery."
    for i in range(1, 70)
)

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

    for i in range(30):
        payload = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{
                "role": "user",
                "content": (
                    PREFIX
                    + f"\n\nUnique request {i}: analyze this incident "
                      "and summarize the likely cause."
                ),
            }],
            "max_tokens": 40,
            "temperature": 0,
        }

        start = time.perf_counter()
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code != 200:
            print("Error response:", r.text)
        r.raise_for_status()
        r.json()
        latencies.append(time.perf_counter() - start)

    p95 = statistics.quantiles(latencies, n=20)[18]

    print(f"target={target}")
    print(f"requests={len(latencies)}")
    print(f"p95_ttft_proxy={p95:.4f}s")
    print(f"min={min(latencies):.4f}s")
    print(f"max={max(latencies):.4f}s")


if __name__ == "__main__":
    main()