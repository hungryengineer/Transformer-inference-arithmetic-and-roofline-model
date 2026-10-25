
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda"
DTYPE = torch.float16

CONTEXT_LENGTHS = [128, 512, 2048]
WARMUP_STEPS = 3
MEASURED_STEPS = 20

assert torch.cuda.is_available(), "CUDA is unavailable"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    trust_remote_code=False,
).to(DEVICE)

model.eval()

print(f"Model: {MODEL_ID}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Parameter dtype: {next(model.parameters()).dtype}")


def make_input(length):
    text = "The quick brown fox jumps over the lazy dog. " * (length + 10)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=length,
    )

    return {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }


@torch.inference_mode()
def prefill(inputs):
    return model(
        **inputs,
        use_cache=True,
        logits_to_keep=1,
    )


@torch.inference_mode()
def decode_steps(cache, attention_mask, steps, measure=False):
    timings = []

    for _ in range(steps):
        # The mask must cover the cached context and the new token.
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=DEVICE,
                ),
            ],
            dim=1,
        )

        if measure:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        output = model(
            input_ids=decode_steps.current_token,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )

        if measure:
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))

        cache = output.past_key_values

        # Greedy token selection for the next decode step.
        decode_steps.current_token = output.logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

    return cache, attention_mask, timings


for context_length in CONTEXT_LENGTHS:
    inputs = make_input(context_length)

    # Warm up the model and CUDA execution path.
    output = prefill(inputs)
    cache = output.past_key_values
    attention_mask = inputs["attention_mask"]

    decode_steps.current_token = output.logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    cache, attention_mask, _ = decode_steps(
        cache,
        attention_mask,
        WARMUP_STEPS,
    )

    del output, cache
    torch.cuda.synchronize()

    # Start the measurement from a fresh prompt and KV cache.
    output = prefill(inputs)
    cache = output.past_key_values
    attention_mask = inputs["attention_mask"]

    decode_steps.current_token = output.logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    torch.cuda.synchronize()

    cache, attention_mask, timings = decode_steps(
        cache,
        attention_mask,
        MEASURED_STEPS,
        measure=True,
    )

    sorted_timings = sorted(timings)
    median_ms = sorted_timings[len(sorted_timings) // 2]
    mean_ms = sum(timings) / len(timings)

    print(f"\nInitial context length: {context_length}")
    print(f"Measured decode steps: {MEASURED_STEPS}")
    print(
        "Decode step samples (ms): "
        f"{[round(t, 3) for t in timings]}"
    )
    print(f"Median decode step: {median_ms:.3f} ms")
    print(f"Mean decode step: {mean_ms:.3f} ms")
    print(f"Median-based throughput: {1000 / median_ms:.2f} tokens/s")

    del inputs, output, cache, attention_mask
    torch.cuda.empty_cache()
