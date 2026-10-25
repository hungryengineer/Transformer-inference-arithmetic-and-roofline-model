
# Project 3: model memory footprint estimator

PARAMETERS = 494_000_000  # approximate parameter count
HIDDEN_SIZE = 896
NUM_LAYERS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 64

BYTES_PER_PARAMETER = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
}


def estimate_weight_bytes(parameters, bytes_per_parameter):
    return parameters * bytes_per_parameter


def estimate_kv_cache_bytes(
    tokens,
    batch_size,
    num_layers,
    num_kv_heads,
    head_dim,
    bytes_per_element,
):
    return (
        tokens
        * batch_size
        * num_layers
        * 2  # key and value
        * num_kv_heads
        * head_dim
        * bytes_per_element
    )


def mib(byte_count):
    return byte_count / (1024 ** 2)


def main():
    tokens = 2048
    batch_size = 1
    cache_bytes_per_element = 2  # FP16/BF16 cache

    print("Weight storage estimates:")
    for dtype, bytes_per_parameter in BYTES_PER_PARAMETER.items():
        size = estimate_weight_bytes(
            PARAMETERS, bytes_per_parameter
        )
        print(f"  {dtype}: {mib(size):.2f} MiB")

    kv_bytes = estimate_kv_cache_bytes(
        tokens=tokens,
        batch_size=batch_size,
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        bytes_per_element=cache_bytes_per_element,
    )

    print(f"\nKV cache at {tokens} tokens, batch {batch_size}:")
    print(f"  {mib(kv_bytes):.2f} MiB")


if __name__ == "__main__":
    main()
