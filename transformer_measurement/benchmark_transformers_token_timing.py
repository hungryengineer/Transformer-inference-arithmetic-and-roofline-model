#!/usr/bin/env python3
"""
Project 3 — Transformers token-arrival instrumentation.

Measures client/process-observed TTFT and TPOT for one request at a time.
The custom streamer timestamps generated token IDs as Transformers emits them.
This is a diagnostic instrumented run; streamer callbacks add some overhead.
It is NOT directly comparable to the earlier non-streaming benchmark.
"""

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "Explain what a Kubernetes pod is in one sentence."
MAX_NEW_TOKENS = 32
WARMUP_RUNS = 2
MEASURED_RUNS = 5
OUTPUT_JSON = "project3_transformers_token_timing.json"


class TimestampStreamer(BaseStreamer):
    """Record host timestamps for generated token IDs; ignore initial prompt IDs."""

    def __init__(self):
        self.timestamps_ns = []
        self.token_ids = []
        self._first_put = True

    def put(self, value):
        # Transformers first calls put() with the complete prompt.
        if self._first_put:
            self._first_put = False
            return

        # In this single-request experiment, each subsequent put is normally
        # one newly generated token. Preserve all IDs if a chunk is emitted.
        ids = value.detach().reshape(-1).cpu().tolist()
        timestamp = time.perf_counter_ns()
        for token_id in ids:
            self.token_ids.append(int(token_id))
            self.timestamps_ns.append(timestamp)

    def end(self):
        pass


def summarize_trial(index, start_ns, end_ns, streamer, generated_ids, tokenizer):
    stamps = streamer.timestamps_ns
    stream_ids = streamer.token_ids
    output_ids = generated_ids.detach().reshape(-1).cpu().tolist()

    # The returned sequence is the authoritative generated-token count.
    # The streamer should have observed the same number of token IDs.
    if len(stream_ids) != len(output_ids):
        raise RuntimeError(
            f"Streamer captured {len(stream_ids)} IDs but generate() returned "
            f"{len(output_ids)} generated IDs. Refusing to calculate TPOT."
        )

    if not stamps:
        raise RuntimeError("No generated-token timestamps were captured.")

    ttft_ms = (stamps[0] - start_ns) / 1e6
    total_generation_ms = (end_ns - start_ns) / 1e6

    if len(stamps) >= 2:
        intervals_ms = [
            (stamps[i] - stamps[i - 1]) / 1e6
            for i in range(1, len(stamps))
        ]
        tpot_ms = (stamps[-1] - stamps[0]) / (len(stamps) - 1) / 1e6
    else:
        intervals_ms = []
        tpot_ms = None

    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return {
        "trial": index,
        "request_start_perf_counter_ns": start_ns,
        "first_token_perf_counter_ns": stamps[0],
        "token_timestamps_perf_counter_ns": stamps,
        "generated_token_ids": output_ids,
        "streamer_token_ids": stream_ids,
        "generated_token_count": len(output_ids),
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "inter_token_intervals_ms": intervals_ms,
        "total_generation_call_ms": total_generation_ms,
        "output_text": text,
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; no measurement was run.")

    print("=" * 72)
    print("Project 3 — Transformers token timestamp experiment")
    print("=" * 72)
    print(f"UTC timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Model: {MODEL_NAME}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Warm-ups: {WARMUP_RUNS}; measured trials: {MEASURED_RUNS}")
    print("Timing: request-call start to host-side streamer token callbacks.")
    print("Caution: streamer/token-ID transfers add instrumentation overhead.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to("cuda")
    model.eval()
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")

    def run_once(index, report=False):
        streamer = TimestampStreamer()
        torch.cuda.synchronize()
        start_ns = time.perf_counter_ns()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                streamer=streamer,
            )
        torch.cuda.synchronize()
        end_ns = time.perf_counter_ns()

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output[0, prompt_len:]
        result = summarize_trial(
            index, start_ns, end_ns, streamer, generated_ids, tokenizer
        )
        if report:
            print(
                f"Trial {index}: TTFT={result['ttft_ms']:.2f} ms, "
                f"TPOT={result['tpot_ms']:.2f} ms/token, "
                f"tokens={result['generated_token_count']}, "
                f"total={result['total_generation_call_ms']:.2f} ms"
            )
            print(f"  Inter-token intervals (ms): "
                  f"{[round(x, 2) for x in result['inter_token_intervals_ms']]}")
            print(f"  Output: {result['output_text']!r}")
        return result

    print("\nWarm-up (excluded from reported measurements)...")
    for i in range(WARMUP_RUNS):
        run_once(-(i + 1))

    print("\nMeasured trials:")
    trials = [run_once(i + 1, report=True) for i in range(MEASURED_RUNS)]

    ttfts = [t["ttft_ms"] for t in trials]
    tpots = [t["tpot_ms"] for t in trials if t["tpot_ms"] is not None]
    payload = {
        "classification": "measurements from this execution",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "precision": "FP16",
        "decoding": "greedy",
        "max_new_tokens": MAX_NEW_TOKENS,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "measurement_boundary": (
            "perf_counter immediately before model.generate; token timestamps "
            "inside custom streamer.put on host; end after CUDA synchronize"
        ),
        "instrumentation_caveat": (
            "Streamer token-ID transfers/callbacks add overhead; these values "
            "are instrumented diagnostics, not directly comparable to the "
            "prior non-streaming benchmark."
        ),
        "median_ttft_ms": statistics.median(ttfts),
        "mean_ttft_ms": statistics.mean(ttfts),
        "median_tpot_ms": statistics.median(tpots) if tpots else None,
        "mean_tpot_ms": statistics.mean(tpots) if tpots else None,
        "trials": trials,
    }

    Path(OUTPUT_JSON).write_text(json.dumps(payload, indent=2))
    print("\n[SUMMARY — measured values from this execution]")
    print(f"Median TTFT: {payload['median_ttft_ms']:.2f} ms")
    print(f"Mean TTFT:   {payload['mean_ttft_ms']:.2f} ms")
    if tpots:
        print(f"Median TPOT: {payload['median_tpot_ms']:.2f} ms/token")
        print(f"Mean TPOT:   {payload['mean_tpot_ms']:.2f} ms/token")
    print(f"Raw timestamps and trial data saved to: {Path(OUTPUT_JSON).resolve()}")


if __name__ == "__main__":
    main()
