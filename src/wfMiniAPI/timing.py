import kernel
import time


print(kernel.get_device_module("cpu"))
print(kernel.get_device_module("gpu"))

print(kernel.list_kernels())
print(kernel.kernel_params("matMulGeneral"))
print(kernel.kernel_params("writeWithMPI"))

n_repeat = 20
kernel.run_kernel("matMulSimple2D", device="cpu", size=8192)
t0 = time.time()
for _ in range(n_repeat):
    kernel.matMulSimple2D(device="cpu", size=8192)
#    kernel.run_kernel("matMulSimple2D", device="cpu", size=8192)
print("took", time.time() - t0)
total_ms = (time.time() - t0) * 1000
avg_ms   = total_ms / n_repeat
print(f"CPU: Total time for {n_repeat} runs: {total_ms} ms")
print(f"CPU: Average per run: {avg_ms} ms")


import cupy as cp
kernel.matMulSimple2D(device="gpu", size=8192)
cp.cuda.Stream.null.synchronize()

start = cp.cuda.Event()
end   = cp.cuda.Event()
start.record()

for _ in range(n_repeat):
    kernel.matMulSimple2D(device="gpu", size=8192)
#    kernel.run_kernel("matMulSimple2D", device="gpu", size=8192)
end.record()
end.synchronize()

total_ms = cp.cuda.get_elapsed_time(start, end)
avg_ms   = total_ms / n_repeat
print(f"GPU: Total time for {n_repeat} runs: {total_ms} ms")
print(f"GPU: Average per run: {avg_ms} ms")
