#!/usr/bin/env python3
"""One-request probe of vLLM AsyncLLM streaming; not a benchmark."""
import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import asyncio
import inspect
import time

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "Explain what a Kubernetes pod is in one sentence."


async def main():
    args = AsyncEngineArgs(
        model=MODEL_NAME,
        dtype="half",
        max_model_len=512,
        gpu_memory_utilization=0.75,
        disable_log_stats=True,
        enable_prefix_caching=False,
        stream_interval=1,
    )
    print("Initializing AsyncLLM...")
    engine = AsyncLLM.from_engine_args(args)
    params = SamplingParams(temperature=0.0, max_tokens=32)

    start_ns = time.perf_counter_ns()
    previous_count = 0
    update_count = 0
    final_token_ids = []
    final_text = None

    try:
        async for request_output in engine.generate(
            PROMPT, params, request_id="project3-async-probe-001"
        ):
            received_ns = time.perf_counter_ns()
            update_count += 1

            if not request_output.outputs:
                print(f"Update {update_count}: no completion outputs")
                continue

            completion = request_output.outputs[0]
            token_ids = list(completion.token_ids)
            count = len(token_ids)
            delta = count - previous_count

            print(f"\nUpdate {update_count}")
            print(f"  elapsed_ms={(received_ns-start_ns)/1e6:.3f}")
            print(f"  token_ids_count={count}")
            print(f"  token_ids={token_ids}")
            print(f"  count_delta={delta}")
            print(f"  request_finished={request_output.finished}")
            print(f"  finish_reason={completion.finish_reason}")

            if delta < 0:
                print("  WARNING: token count decreased.")
            elif delta == 0:
                print("  NOTE: no additional token IDs in this update.")

            previous_count = count
            final_token_ids = token_ids
            final_text = completion.text

        print("\n=== PROBE SUMMARY ===")
        print("Async generator updates:", update_count)
        print("Final token ID count:", len(final_token_ids))
        print("Final output:", repr(final_text))
        print("Total wall time ms:",
              round((time.perf_counter_ns()-start_ns)/1e6, 3))
        print("This is an API behavior probe, not a TTFT/TPOT benchmark.")
    finally:
        print("\nShutting down AsyncLLM...")
        result = engine.shutdown()
        if inspect.isawaitable(result):
            await result
        print("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
