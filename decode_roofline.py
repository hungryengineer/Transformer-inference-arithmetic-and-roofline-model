
# Project 3: first-pass decode roofline estimator
# All bandwidths use decimal GB/s; FLOPs count multiply + add separately.

NUM_PARAMETERS = 494_000_000
BYTES_PER_WEIGHT = 2          # FP16/BF16
NUM_LAYERS = 24
HIDDEN = 896
INTERMEDIATE = 4864
VOCAB_SIZE = 151936
NUM_KV_HEADS = 2
HEAD_DIM = 64
BYTES_PER_KV_ELEMENT = 2

# Measured on this machine in our microbenchmarks.
COMPUTE_TFLOPS = 27.95
BANDWIDTH_GB_S = 320.99


def estimate_decode(context_tokens):
    # Major linear-layer FLOPs per token, per Transformer block:
    # Q, K, V, output projection + two up projections + down projection.
    
    kv_width = NUM_KV_HEADS * HEAD_DIM

    projection_flops = (
        4 * HIDDEN * HIDDEN
        + 4 * HIDDEN * kv_width
        + 6 * HIDDEN * INTERMEDIATE
    )


    # One-token decode attention:
    # QK^T and attention-weighted V, across all query heads.
    attention_flops_per_layer = 4 * context_tokens * HIDDEN

    # Output projection to vocabulary; weights are tied with embeddings.
    lm_head_flops = 2 * HIDDEN * VOCAB_SIZE

    total_flops = (
        NUM_LAYERS
        * (projection_flops + attention_flops_per_layer)
        + lm_head_flops
    )

    # Simplified assumption: all model weights are read for this token.
    weight_bytes = NUM_PARAMETERS * BYTES_PER_WEIGHT

    # K and V, for every cached token and every layer.
    kv_bytes_per_token = (
        NUM_LAYERS
        * 2
        * NUM_KV_HEADS
        * HEAD_DIM
        * BYTES_PER_KV_ELEMENT
    )
    kv_bytes = context_tokens * kv_bytes_per_token

    total_bytes = weight_bytes + kv_bytes

    # Separate compute and bandwidth lower bounds.
    compute_seconds = total_flops / (COMPUTE_TFLOPS * 1e12)
    memory_seconds = total_bytes / (BANDWIDTH_GB_S * 1e9)
    roofline_seconds = max(compute_seconds, memory_seconds)

    arithmetic_intensity = total_flops / total_bytes

    return {
        "flops": total_flops,
        "bytes": total_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "compute_ms": compute_seconds * 1000,
        "memory_ms": memory_seconds * 1000,
        "roofline_ms": roofline_seconds * 1000,
    }


if __name__ == "__main__":
    for context in (128, 512, 2048, 8192):
        result = estimate_decode(context)

        print(f"\nContext: {context:,} tokens")
        print(f"FLOPs: {result['flops'] / 1e9:.3f} GFLOPs")
        print(f"Estimated bytes: {result['bytes'] / 1e9:.3f} GB")
        print(
            "Arithmetic intensity: "
            f"{result['arithmetic_intensity']:.3f} FLOPs/byte"
        )
        print(f"Compute lower bound: {result['compute_ms']:.4f} ms")
        print(f"Memory lower bound: {result['memory_ms']:.4f} ms")
        print(f"Roofline lower bound: {result['roofline_ms']:.4f} ms")
