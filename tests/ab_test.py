import concurrent.futures
import json
import statistics
import sys
import time

import requests


ENDPOINTS = {
    "rr": "http://localhost:8005/v1/chat/completions",
    "prefix": "http://192.168.1.40:8000/v1/chat/completions",
}

PROM_URL = "http://localhost:9090"

# ~1500-token-ish shared prefix.
PREFIXES = []
for p_idx in range(8):
    prefix = " ".join(
        f"Shared incident context {p_idx} section {i}: "
        "production Kubernetes inference systems require careful "
        "capacity planning, observability, scheduling, reliability, "
        "networking, and recovery."
        for i in range(1, 50)
    )
    PREFIXES.append(prefix)


def prometheus_value(metric: str) -> float:
    r = requests.get(
        f"{PROM_URL}/api/v1/query",
        params={"query": metric},
        timeout=5,
    )
    r.raise_for_status()
    results = r.json()["data"]["result"]

    if not results:
        return 0.0

    return sum(float(item["value"][1]) for item in results)


def cache_snapshot():
    return {
        "queries": prometheus_value(
            "sum(vllm:prefix_cache_queries_total)"
        ),
        "hits": prometheus_value(
            "sum(vllm:prefix_cache_hits_total)"
        ),
    }


def measure_ttft(url: str, request_id: int) -> float:
    prefix = PREFIXES[request_id % len(PREFIXES)]

    payload = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": (
                    prefix
                    + f"\n\nUnique request {request_id}: "
                      "analyze this incident and summarize the likely cause."
                ),
            }
        ],
        "max_tokens": 40,
        "temperature": 0,
        "stream": True,
    }

    start = time.perf_counter()

    with requests.post(
        url,
        json=payload,
        timeout=30,
        stream=True,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            data = line[6:]

            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content")

            # First actual generated content token/chunk.
            if content:
                return time.perf_counter() - start

    raise RuntimeError("No streamed content received")


def p95(values):
    if len(values) < 2:
        return values[0]
    return statistics.quantiles(values, n=20)[18]


def run(target: str):
    url = ENDPOINTS[target]

    print(f"\n=== {target.upper()} ===")

    before = cache_snapshot()

    latencies = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=30
    ) as executor:
        futures = [
            executor.submit(measure_ttft, url, i)
            for i in range(800)
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception as exc:
                print(f"request failed: {exc}")

    after = cache_snapshot()

    if not latencies:
        raise RuntimeError("No successful requests")

    query_delta = after["queries"] - before["queries"]
    hit_delta = after["hits"] - before["hits"]

    hit_rate = (
        hit_delta / query_delta
        if query_delta > 0
        else 0.0
    )

    print(f"successful_requests={len(latencies)}")
    print(f"p50_ttft={statistics.median(latencies):.4f}s")
    print(f"p95_ttft={p95(latencies):.4f}s")
    print(f"min_ttft={min(latencies):.4f}s")
    print(f"max_ttft={max(latencies):.4f}s")
    print(f"cache_queries_delta={query_delta:.0f}")
    print(f"cache_hits_delta={hit_delta:.0f}")
    print(f"cache_hit_rate={hit_rate * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ENDPOINTS:
        print("Usage: python3 ab_test_ttft.py [rr|prefix]")
        sys.exit(1)

    run(sys.argv[1])