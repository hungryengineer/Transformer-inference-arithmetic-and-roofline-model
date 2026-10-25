
# Project 3: first-pass prefill FLOPs estimator
# Model: Qwen2.5-0.5B
#
# Estimates major matrix-multiplication FLOPs.
# Assumptions:
# - Causal attention skips the masked upper triangle.
# - Output logits are calculated only for the final prompt token.
# - One multiplication + one addition = 2 FLOPs.

HIDDEN = 896
INTERMEDIATE = 4864
NUM_LAYERS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 64
VOCAB_SIZE = 151936


def estimate_prefill_flops(sequence_length):
    h = HIDDEN
    ffn = INTERMEDIATE
    kv_width = NUM_KV_HEADS * HEAD_DIM
    s = sequence_length

    # Linear projections per token, per Transformer layer:
    # Q and output projections
    # K and V projections
    # SwiGLU gate, up, and down projections
    linear_flops_per_token = (
        4 * h * h
        + 4 * h * kv_width
        + 6 * h * ffn
    )

    # Approximate causal attention FLOPs per layer.
    # Uses the triangular-attention approximation:
    # QK^T + attention-weighted V ≈ 2 * S^2 * H.
    attention_flops_per_layer = 2 * (s ** 2) * h

    transformer_flops = NUM_LAYERS * (
        s * linear_flops_per_token
        + attention_flops_per_layer
    )

    # Next-token inference generally needs logits only
    # for the final prompt position.
    output_head_flops = 2 * h * VOCAB_SIZE

    return transformer_flops + output_head_flops


if __name__ == "__main__":
    for tokens in (128, 512, 2048, 8192):
        flops = estimate_prefill_flops(tokens)

        print(f"\nPrompt length: {tokens:,} tokens")
        print(f"Prefill FLOPs: {flops / 1e9:.3f} GFLOPs")
        print(
            "FLOPs per prompt token: "
            f"{flops / tokens / 1e9:.3f} GFLOPs"
        )
