
import torch

assert torch.cuda.is_available(), "CUDA is not available"

DEVICE = "cuda"
DTYPE = torch.float16
N = 4096
WARMUP_ITERATIONS = 5
MEASURED_ITERATIONS = 30

# A and B: 4096 x 4096 FP16 matrices.
a = torch.randn((N, N), device=DEVICE, dtype=DTYPE)
b = torch.randn((N, N), device=DEVICE, dtype=DTYPE)

# Warm up the GPU and matmul implementation.
for _ in range(WARMUP_ITERATIONS):
    c = torch.matmul(a, b)

torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

for _ in range(MEASURED_ITERATIONS):
    c = torch.matmul(a, b)

end.record()
torch.cuda.synchronize()

elapsed_seconds = start.elapsed_time(end) / 1000.0
average_seconds = elapsed_seconds / MEASURED_ITERATIONS

# One dense matrix multiplication:
# N^3 multiplications + N^3 additions, approximately.
flops_per_matmul = 2 * (N ** 3)

achieved_tflops = (
    flops_per_matmul * MEASURED_ITERATIONS
    / elapsed_seconds
    / 1e12
)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Data type: {DTYPE}")
print(f"Matrix shape: {N} x {N}")
print(f"Average matmul time: {average_seconds * 1000:.3f} ms")
print(f"Estimated achieved throughput: {achieved_tflops:.2f} TFLOPs/s")
print("FLOPs count multiplication and addition separately.")
