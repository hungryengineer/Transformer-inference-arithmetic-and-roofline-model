#!/usr/bin/env python3
"""
Project 3 — controlled single-request generation benchmark.

Runs Hugging Face Transformers and vLLM sequentially with:
- the same model and prompt
- FP16 weights
- greedy decoding
- the same maximum number of new tokens
- warm-up requests excluded from timing
- repeated timed requests
- median latency and observed generated tokens/second

This is a single-request microbenchmark, not a concurrency/load test.
The engines run sequentially, so GPU clocks, temperature, and background activity
may differ between the two measurement windows. Results are observations, not
proof that one engine is universally faster.

Run in one Python environment with compatible torch, transformers, and vllm:
  VLLM_USE_FLASHINFER_SAMPLER=0 uv run python /path/to/benchmark_transformers_vllm.py
"""

# Workaround required by the previously observed FlashInfer sampler startup failure.
# Set this before importing vLLM.
import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import gc
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "Explain what a Kubernetes pod is in one sentence."
MAX_NEW_TOKENS = 32
MAX_MODEL_LEN = 512
GPU_MEMORY_UTILIZATION = 0.75
WARMUP_RUNS = 2
MEASURED_RUNS = 5
OUTPUT_JSON = "project3_single_request_benchmark.json"


def summarize(name, latencies_s, token_counts, outputs):
    """Summarize measured request wall times; token counts come from each output."""
    median_s = statistics.median(latencies_s)
    mean_s = statistics.mean(latencies_s)
    # Per-run rate is generated tokens divided by that run's measured wall time.
    rates = [
        tokens / elapsed if elapsed > 0 else 0.0
        for tokens, elapsed in zip(token_counts, latencies_s)
    ]
    median_rate = statistics.median(rates)

    return {
        "engine": name,
        "latency_seconds_each_run": latencies_s,
        "generated_tokens_each_run": token_counts,
        "median_latency_seconds": median_s,
        "mean_latency_seconds": mean_s,
        "tokens_per_second_each_run": rates,
        "median_observed_tokens_per_second": median_rate,
        "sample_output": outputs[-1] if outputs else None,
    }


def main():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from vllm import LLM, SamplingParams
    except Exception as exc:
        print("DEPENDENCY/IMPORT ERROR")
        print(f"{type(exc).__name__}: {exc}")
        print(
            "\nThis script needs torch, transformers, and vllm in the SAME "
            "Python environment. No benchmark was run."
        )
        sys.exit(2)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable. No benchmark was run.")
        sys.exit(2)

    print("=" * 76)
    print("Project 3 — controlled single-request generation benchmark")
    print("=" * 76)
    print(f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Precision: FP16")
    print(f"Decoding: greedy (Transformers do_sample=False; vLLM temperature=0)")
    print(f"Maximum new tokens: {MAX_NEW_TOKENS}")
    print(f"Warm-up runs per engine: {WARMUP_RUNS}")
    print(f"Measured runs per engine: {MEASURED_RUNS}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"FlashInfer sampler workaround: {os.environ['VLLM_USE_FLASHINFER_SAMPLER']}")
    print("\nTiming boundary: generation call only; model initialization excluded.")
    print("Note: engine windows are sequential, not perfectly thermally isolated.")

    results = []
    sample_texts = {}

    # ------------------------------------------------------------------
    # Benchmark A: Hugging Face Transformers
    # Keep the model resident through warm-up and all measured trials.
    # ------------------------------------------------------------------
    print("\n[A] Hugging Face Transformers — loading model (not timed)")
    hf_model = None
    hf_tokenizer = None
    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
        ).to("cuda")
        hf_model.eval()
        inputs = hf_tokenizer(PROMPT, return_tensors="pt").to("cuda")

        def run_transformers_once():
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = hf_model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            prompt_len = inputs["input_ids"].shape[1]
            new_ids = output_ids[0, prompt_len:]
            text = hf_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            return elapsed, int(new_ids.numel()), text

        print(f"[A] Running {WARMUP_RUNS} warm-up request(s)...")
        for _ in range(WARMUP_RUNS):
            run_transformers_once()

        hf_latencies = []
        hf_tokens = []
        hf_outputs = []
        print(f"[A] Running {MEASURED_RUNS} measured request(s)...")
        for i in range(MEASURED_RUNS):
            elapsed, token_count, text = run_transformers_once()
            hf_latencies.append(elapsed)
            hf_tokens.append(token_count)
            hf_outputs.append(text)
            print(
                f"  Trial {i + 1}: {elapsed * 1000:.2f} ms, "
                f"{token_count} generated tokens, "
                f"{token_count / elapsed if elapsed else 0:.2f} tokens/s"
            )

        results.append(
            summarize("Transformers", hf_latencies, hf_tokens, hf_outputs)
        )
        sample_texts["Transformers"] = hf_outputs[-1]
        print(f"[A] Sample output: {hf_outputs[-1]!r}")
    except Exception as exc:
        print(f"[A] FAILED: {type(exc).__name__}: {exc}")
        print("No valid Transformers benchmark result was produced.")
    finally:
        if hf_model is not None:
            del hf_model
        if hf_tokenizer is not None:
            del hf_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Benchmark B: vLLM
    # Engine/model initialization and graph capture occur before timing.
    # The timed region contains the generate request call only.
    # ------------------------------------------------------------------
    print("\n[B] vLLM — initializing engine (not timed)")
    llm = None
    try:
        llm = LLM(
            model=MODEL_NAME,
            dtype="half",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        )
        params = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_NEW_TOKENS,
        )

        def run_vllm_once():
            start = time.perf_counter()
            outputs = llm.generate([PROMPT], params)
            elapsed = time.perf_counter() - start
            request_output = outputs[0].outputs[0]
            text = request_output.text.strip()
            token_count = len(request_output.token_ids)
            return elapsed, token_count, text

        print(f"[B] Running {WARMUP_RUNS} warm-up request(s)...")
        for _ in range(WARMUP_RUNS):
            run_vllm_once()

        vllm_latencies = []
        vllm_tokens = []
        vllm_outputs = []
        print(f"[B] Running {MEASURED_RUNS} measured request(s)...")
        for i in range(MEASURED_RUNS):
            elapsed, token_count, text = run_vllm_once()
            vllm_latencies.append(elapsed)
            vllm_tokens.append(token_count)
            vllm_outputs.append(text)
            print(
                f"  Trial {i + 1}: {elapsed * 1000:.2f} ms, "
                f"{token_count} generated tokens, "
                f"{token_count / elapsed if elapsed else 0:.2f} tokens/s"
            )

        results.append(summarize("vLLM", vllm_latencies, vllm_tokens, vllm_outputs))
        sample_texts["vLLM"] = vllm_outputs[-1]
        print(f"[B] Sample output: {vllm_outputs[-1]!r}")
    except Exception as exc:
        print(f"[B] FAILED: {type(exc).__name__}: {exc}")
        print("No valid vLLM benchmark result was produced.")
    finally:
        if llm is not None:
            del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Report only results that were actually measured in this run.
    # JSON output contains the raw trial values and summary statistics.
    # ------------------------------------------------------------------
    print("\n[SUMMARY — measured values from this run]")
    if not results:
        print("No valid benchmark results were produced.")
        sys.exit(1)

    for result in results:
        print(f"\n{result['engine']}:")
        print(f"  Median latency: {result['median_latency_seconds'] * 1000:.2f} ms")
        print(f"  Mean latency:   {result['mean_latency_seconds'] * 1000:.2f} ms")
        print(
            "  Median observed output rate: "
            f"{result['median_observed_tokens_per_second']:.2f} tokens/s"
        )
        print(f"  Trial latencies (ms): {[round(x * 1000, 2) for x in result['latency_seconds_each_run']]}")

    if len(results) == 2:
        same = sample_texts["Transformers"] == sample_texts["vLLM"]
        print(f"\nLast-trial exact output match: {same}")
        print("This is a one-prompt diagnostic, not a general correctness proof.")

    payload = {
        "classification": "measured single-request benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "precision": "FP16",
        "decoding": "greedy",
        "max_new_tokens": MAX_NEW_TOKENS,
        "warmup_runs_per_engine": WARMUP_RUNS,
        "measured_runs_per_engine": MEASURED_RUNS,
        "gpu": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "flashinfer_sampler_workaround": os.environ["VLLM_USE_FLASHINFER_SAMPLER"],
        "results": results,
        "output_texts": sample_texts,
        "caveats": [
            "Single-request microbenchmark only; no concurrency/load test.",
            "Model initialization excluded from generation timing.",
            "Engine windows run sequentially; GPU clocks/temperature/background activity may differ.",
            "Tokens/second is generated token count divided by measured generation-call wall time.",
            "One prompt does not establish general correctness or universal engine performance.",
        ],
    }
    output_path = Path(OUTPUT_JSON)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nRaw trials and summary saved to: {output_path.resolve()}")
    print("Classification: measurements from this execution only; no old values are reused.")


if __name__ == "__main__":
    main()
