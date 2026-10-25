
import torch

assert torch.cuda.is_available(), "CUDA is not available"

DEVICE = "cuda"
NUM_ELEMENTS = 32 * 1024 * 1024  # 32M FP32 elements
WARMUP_ITERATIONS = 5
MEASURED_ITERATIONS = 30

# Each tensor is 128 MiB. Two tensors use 256 MiB total.
src = torch.randn(NUM_ELEMENTS, device=DEVICE, dtype=torch.float32)
dst = torch.empty_like(src)

torch.cuda.synchronize()

# Warm up the copy operation before measuring.
for _ in range(WARMUP_ITERATIONS):
    dst.copy_(src)

torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

for _ in range(MEASURED_ITERATIONS):
    dst.copy_(src)

end.record()
torch.cuda.synchronize()

elapsed_seconds = start.elapsed_time(end) / 1000.0

bytes_per_copy = NUM_ELEMENTS * src.element_size()

# Count both source reads and destination writes.
total_bytes = (
    2 * bytes_per_copy * MEASURED_ITERATIONS
)

bandwidth_gb_s = total_bytes / elapsed_seconds / 1e9
average_copy_ms = elapsed_seconds * 1000 / MEASURED_ITERATIONS

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Elements per copy: {NUM_ELEMENTS:,}")
print(f"Bytes per tensor: {bytes_per_copy / (1024**2):.2f} MiB")
print(f"Measured iterations: {MEASURED_ITERATIONS}")
print(f"Average copy time: {average_copy_ms:.3f} ms")
print(f"Effective bandwidth: {bandwidth_gb_s:.2f} GB/s")
print("Bandwidth counts source reads + destination writes.")
