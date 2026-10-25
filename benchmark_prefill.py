
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda"
DTYPE = torch.float16

PROMPT_LENGTHS = [128, 512, 2048]
WARMUP_ITERATIONS = 2
MEASURED_ITERATIONS = 5

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
print(
    "Model memory allocated: "
    f"{torch.cuda.memory_allocated() / (1024**2):.2f} MiB"
)


def make_input(length):
    # Repeat ordinary text to obtain a reproducible token count.
    text = "The quick brown fox jumps over the lazy dog. " * (length + 10)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=length,
        padding="max_length",
    )
    return {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }


def timed_forward(inputs):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.inference_mode():
        try:
            # Request only the final-position logits if supported.
            output = model(
                **inputs,
                use_cache=True,
                logits_to_keep=1,
            )
        except TypeError:
            # Compatibility fallback for model versions without
            # the logits_to_keep argument.
            output = model(**inputs, use_cache=True)
    end.record()
    torch.cuda.synchronize()

    # Keep output alive until the timing completes.
    assert output.logits is not None
    return start.elapsed_time(end)


for length in PROMPT_LENGTHS:
    inputs = make_input(length)

    for _ in range(WARMUP_ITERATIONS):
        with torch.inference_mode():
            try:
                model(**inputs, use_cache=True, logits_to_keep=1)
            except TypeError:
                model(**inputs, use_cache=True)
        torch.cuda.synchronize()

    timings = [
        timed_forward(inputs)
        for _ in range(MEASURED_ITERATIONS)
    ]

    timings.sort()
    median_ms = timings[len(timings) // 2]

    print(f"\nPrompt length: {length}")
    print(f"Forward latency samples (ms): {timings}")
    print(f"Median prefill forward latency: {median_ms:.3f} ms")
    print(
        "Median time per prompt token: "
        f"{median_ms / length:.4f} ms"
    )
