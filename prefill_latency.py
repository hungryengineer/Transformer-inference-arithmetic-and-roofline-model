
# Project 3: first-pass prefill roofline estimator
# Qwen2.5-0.5B, FP16/BF16 assumptions.

PARAMETERS = 494_000_000
HIDDEN = 896
INTERMEDIATE = 4864
NUM_LAYERS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 64
VOCAB_SIZE = 151936

BYTES_PER_WEIGHT = 2
BYTES_PER_KV_ELEMENT = 2

# Empirical reference measurements from this machine.
COMPUTE_TFLOPS = 27.95
BANDWIDTH_GB_S = 320.99


def estimate_prefill_flops(sequence_length):
    h = HIDDEN
    ffn = INTERMEDIATE
    kv_width = NUM_KV_HEADS * HEAD_DIM
    s = sequence_length

    linear_flops_per_token = (
        4 * h * h
        + 4 * h * kv_width
        + 6 * h * ffn
    )

    # Approximate causal attention work.
    attention_flops_per_layer = 2 * s * s * h

    transformer_flops = NUM_LAYERS * (
        s * linear_flops_per_token
        + attention_flops_per_layer
    )

    # Only final-position logits for next-token generation.
    output_head_flops = 2 * h * VOCAB_SIZE

    return transformer_flops + output_head_flops



def estimate_prefill_bytes(sequence_length):
    # First-order approximation: count all model parameters
    # once at the selected weight precision.
    # The tied embedding/output-head matrix is included once.
    model_weight_bytes = PARAMETERS * BYTES_PER_WEIGHT

    # KV cache is populated during prefill.
    kv_bytes_per_token = (
        NUM_LAYERS
        * 2  # Key and value
        * NUM_KV_HEADS
        * HEAD_DIM
        * BYTES_PER_KV_ELEMENT
    )
    kv_write_bytes = sequence_length * kv_bytes_per_token

    return model_weight_bytes + kv_write_bytes



def estimate_latency(sequence_length):
    flops = estimate_prefill_flops(sequence_length)
    byte_count = estimate_prefill_bytes(sequence_length)

    compute_seconds = flops / (COMPUTE_TFLOPS * 1e12)
    memory_seconds = byte_count / (BANDWIDTH_GB_S * 1e9)

    return {
        "flops": flops,
        "bytes": byte_count,
        "compute_ms": compute_seconds * 1000,
        "memory_ms": memory_seconds * 1000,
        "roofline_ms": max(compute_seconds, memory_seconds) * 1000,
    }


if __name__ == "__main__":
    for tokens in (128, 512, 2048, 8192):
        result = estimate_latency(tokens)

        print(f"\nPrompt length: {tokens:,} tokens")
        print(f"FLOPs: {result['flops'] / 1e9:.3f} GFLOPs")
        print(f"Estimated bytes: {result['bytes'] / 1e6:.2f} MB")
        print(f"Compute lower bound: {result['compute_ms']:.3f} ms")
        print(f"Memory lower bound: {result['memory_ms']:.3f} ms")
        print(f"Roofline lower bound: {result['roofline_ms']:.3f} ms")
