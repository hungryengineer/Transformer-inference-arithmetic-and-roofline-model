import concurrent.futures
import time
import requests

URL = "http://192.168.1.40:8000/v1/chat/completions"

PAYLOAD = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "Explain production Kubernetes architecture, scheduling, networking, observability, scaling, and failure recovery."
        }
    ],
    "max_tokens": 100,
    "temperature": 0,
}


def request(i):
    start = time.perf_counter()

    r = requests.post(
        URL,
        json=PAYLOAD,
        timeout=30,
    )
    r.raise_for_status()

    elapsed = time.perf_counter() - start
    data = r.json()

    return {
        "id": i,
        "latency": elapsed,
        "tokens": data["usage"]["completion_tokens"],
    }


def run(concurrency, total_requests):
    start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        results = list(
            executor.map(request, range(total_requests))
        )

    elapsed = time.perf_counter() - start

    total_tokens = sum(r["tokens"] for r in results)
    latencies = [r["latency"] for r in results]

    print(f"\nConcurrency: {concurrency}")
    print(f"Requests:    {total_requests}")
    print(f"Wall time:   {elapsed:.3f}s")
    print(f"Tokens:      {total_tokens}")
    print(f"Throughput:  {total_tokens / elapsed:.1f} tok/s")
    print(f"Avg latency: {sum(latencies) / len(latencies):.3f}s")
    print(f"Max latency: {max(latencies):.3f}s")


run(concurrency=75, total_requests=2000)
