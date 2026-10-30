#!/usr/bin/env python3
"""
Controlled vLLM token-timing benchmark using AsyncLLM incremental outputs.

Measures host-observed timing:
- TTFT: first incremental token update minus generation-call start.
- TPOT: average interval between subsequent token updates.
- Per-token intervals and total generation-call wall time.

These are host-observed async-engine timings, NOT isolated GPU kernel timings
and NOT end-to-end HTTP/server TTFT.

The benchmark performs 2 warmups followed by 5 measured requests.
"""

import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import asyncio
import json
import statistics
import time
from pathlib import Path

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


MODEL = "Qwen/Qwen2.5-0.5B"
PROMPT = "Explain what a Kubernetes pod is in one sentence."
MAX_TOKENS = 32
WARMUP_REQUESTS = 2
MEASURED_REQUESTS = 5

DTYPE = "half"
MAX_MODEL_LEN = 512
GPU_MEMORY_UTILIZATION = 0.75

OUTPUT_FILE = Path("vllm_token_timing_results.json")


def median_or_none(values):
    return statistics.median(values) if values else None


def mean_or_none(values):
    return statistics.mean(values) if values else None


async def run_one_request(engine, request_number, is_warmup):
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )

    request_id = f"project3-token-benchmark-{request_number:02d}"
    start_ns = time.perf_counter_ns()

    update_times_ms = []
    cumulative_token_counts = []
    previous_count = 0
    final_text = ""
    final_token_ids = []
    finished = False
    finish_reason = None

    async for request_output in engine.generate(
        PROMPT,
        sampling_params,
        request_id=request_id,
    ):
        observed_ns = time.perf_counter_ns()
        completion = request_output.outputs[0]

        token_ids = list(completion.token_ids)
        current_count = len(token_ids)
        delta = current_count - previous_count

        if delta < 0:
            raise RuntimeError(
                f"Token count decreased: {previous_count} -> {current_count}"
            )

        # The probe established cumulative token_ids and one new token per
        # update for this configuration. Fail closed if that changes.
        if delta > 1:
            raise RuntimeError(
                f"Received {delta} new tokens in one update. "
                "Cannot assign individual token arrival times reliably."
            )

        if delta == 1:
            update_times_ms.append((observed_ns - start_ns) / 1_000_000)
            cumulative_token_counts.append(current_count)

        previous_count = current_count
        final_text = completion.text
        final_token_ids = token_ids
        finished = bool(request_output.finished)
        finish_reason = completion.finish_reason

    end_ns = time.perf_counter_ns()
    total_wall_ms = (end_ns - start_ns) / 1_000_000

    if not finished:
        raise RuntimeError(f"Request {request_id} ended without finished=True")

    if len(update_times_ms) != len(final_token_ids):
        raise RuntimeError(
            "Observed update count does not match final token count: "
            f"{len(update_times_ms)} vs {len(final_token_ids)}"
        )

    if not update_times_ms:
        raise RuntimeError(f"Request {request_id} produced no token updates")

    ttft_ms = update_times_ms[0]
    intervals_ms = [
        update_times_ms[i] - update_times_ms[i - 1]
        for i in range(1, len(update_times_ms))
    ]
    tpot_ms = mean_or_none(intervals_ms)
    output_rate_tps = (
        1000.0 / tpot_ms if tpot_ms is not None and tpot_ms > 0 else None
    )

    result = {
        "request_id": request_id,
        "warmup": is_warmup,
        "prompt": PROMPT,
        "output_text": final_text,
        "token_ids": final_token_ids,
        "token_count": len(final_token_ids),
        "update_count": len(update_times_ms),
        "ttft_host_observed_ms": ttft_ms,
        "tpot_host_observed_ms": tpot_ms,
        "inter_token_intervals_ms": intervals_ms,
        "total_generation_call_wall_ms": total_wall_ms,
        "output_rate_from_tpot_tokens_per_second": output_rate_tps,
        "finish_reason": finish_reason,
    }

    return result


async def main():
    engine_args = AsyncEngineArgs(
        model=MODEL,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_log_stats=True,
        enable_prefix_caching=False,
        stream_interval=1,
    )

    print("Initializing AsyncLLM...")
    engine = AsyncLLM.from_engine_args(engine_args)

    results = []
    try:
        total_requests = WARMUP_REQUESTS + MEASURED_REQUESTS

        for request_number in range(1, total_requests + 1):
            is_warmup = request_number <= WARMUP_REQUESTS
            label = "WARMUP" if is_warmup else "MEASURED"

            result = await run_one_request(
                engine,
                request_number,
                is_warmup,
            )

            results.append(result)

            print(f"\n[{label}] Request {request_number}")
            print(f"  text: {result['output_text']!r}")
            print(f"  tokens: {result['token_count']}")
            print(f"  updates: {result['update_count']}")
            print(
                "  TTFT (host-observed): "
                f"{result['ttft_host_observed_ms']:.3f} ms"
            )
            if result["tpot_host_observed_ms"] is not None:
                print(
                    "  TPOT (host-observed): "
                    f"{result['tpot_host_observed_ms']:.3f} ms"
                )
            print(
                "  total generation-call wall time: "
                f"{result['total_generation_call_wall_ms']:.3f} ms"
            )
            print(
                "  finish_reason: "
                f"{result['finish_reason']}"
            )

    finally:
        print("\nShutting down AsyncLLM...")
        shutdown_result = engine.shutdown()
        if asyncio.iscoroutine(shutdown_result):
            await shutdown_result
        print("Shutdown complete.")

    measured = [r for r in results if not r["warmup"]]

    # All requests must return the same token count and text for this
    # deterministic, single-prompt greedy benchmark.
    token_counts = {r["token_count"] for r in measured}
    output_texts = {r["output_text"] for r in measured}
    if len(token_counts) != 1 or len(output_texts) != 1:
        raise RuntimeError(
            "Measured requests were not output-consistent. "
            f"Token counts={sorted(token_counts)}, "
            f"distinct outputs={len(output_texts)}"
        )

    ttfts = [r["ttft_host_observed_ms"] for r in measured]
    tpots = [
        r["tpot_host_observed_ms"]
        for r in measured
        if r["tpot_host_observed_ms"] is not None
    ]
    totals = [r["total_generation_call_wall_ms"] for r in measured]
    rates = [
        r["output_rate_from_tpot_tokens_per_second"]
        for r in measured
        if r["output_rate_from_tpot_tokens_per_second"] is not None
    ]

    summary = {
        "model": MODEL,
        "dtype": DTYPE,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "warmup_requests": WARMUP_REQUESTS,
        "measured_requests": MEASURED_REQUESTS,
        "timing_boundary": (
            "perf_counter_ns immediately before AsyncLLM.generate() "
            "through host receipt of incremental RequestOutput updates; "
            "not HTTP end-to-end and not GPU-only timing"
        ),
        "token_update_semantics": (
            "Cumulative token_ids; this script requires exactly one new "
            "token per yielded update."
        ),
        "validation": {
            "same_token_count": len(token_counts) == 1,
            "same_output_text": len(output_texts) == 1,
            "measured_token_count": measured[0]["token_count"],
            "measured_output_text": measured[0]["output_text"],
        },
        "summary_metrics": {
            "ttft_host_observed_ms_median": median_or_none(ttfts),
            "ttft_host_observed_ms_mean": mean_or_none(ttfts),
            "tpot_host_observed_ms_median": median_or_none(tpots),
            "tpot_host_observed_ms_mean": mean_or_none(tpots),
            "total_generation_call_wall_ms_median": median_or_none(totals),
            "total_generation_call_wall_ms_mean": mean_or_none(totals),
            "output_rate_from_tpot_tokens_per_second_median": median_or_none(rates),
        },
        "requests": results,
    }

    OUTPUT_FILE.write_text(json.dumps(summary, indent=2))
    print("\n=== MEASURED SUMMARY (warmups excluded) ===")
    print(f"TTFT median: {summary['summary_metrics']['ttft_host_observed_ms_median']:.3f} ms")
    print(f"TTFT mean:   {summary['summary_metrics']['ttft_host_observed_ms_mean']:.3f} ms")
    print(f"TPOT median: {summary['summary_metrics']['tpot_host_observed_ms_median']:.3f} ms")
    print(f"TPOT mean:   {summary['summary_metrics']['tpot_host_observed_ms_mean']:.3f} ms")
    print(f"Total median:{summary['summary_metrics']['total_generation_call_wall_ms_median']:.3f} ms")
    print(f"Total mean:  {summary['summary_metrics']['total_generation_call_wall_ms_mean']:.3f} ms")
    print(
        "Rate from TPOT median: "
        f"{summary['summary_metrics']['output_rate_from_tpot_tokens_per_second_median']:.2f} tok/s"
    )
    print(f"\nFull raw results saved to: {OUTPUT_FILE.resolve()}")
    print("Timing is host-observed async output timing, not GPU-only timing.")


if __name__ == "__main__":
    asyncio.run(main())
