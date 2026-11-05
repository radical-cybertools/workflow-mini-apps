import inspect
from collections import OrderedDict
import time
import numpy as np

_KERNELS = OrderedDict()

class KernelSpec:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self.sig  = inspect.signature(func)
        self.doc  = inspect.getdoc(func) or ""

    def params(self):
        out = {}
        for name, param in self.sig.parameters.items():
            default = param.default if param.default is not inspect._empty else None
            ann     = param.annotation if param.annotation is not inspect._empty else None
            out[name] = {
                'default':     default,
                'annotation':  ann,
            }
        return out

def annotate_kernel(func):
    _KERNELS[func.__name__] = KernelSpec(func)
    return func

def list_kernels():
    return list(_KERNELS)

def kernel_params(name):
    return _KERNELS[name].params()

def run_kernel(name, **kwargs):
    func = _KERNELS[name].func
    out = func(**kwargs)
    return out

def time_kernel(name, device, n_warmup=3, n_repeat=20, **kwargs):
    # CPU timing
    if device.lower() == "cpu":
        for _ in range(n_warmup):
            run_kernel(name, device=device, **kwargs)
        run_times = []
        for _ in range(n_repeat):
            t0 = time.time()
            run_kernel(name, device=device, **kwargs)
            t1 = time.time()
            run_times.append((t1 - t0) * 1000)

        total_ms = sum(run_times)
        avg_ms   = total_ms / n_repeat
        std_ms   = np.std(run_times)
        print(f"CPU: Total {n_repeat} runs: {total_ms:.2f} ms")
        print(f"CPU: Avg per run: {avg_ms:.2f} \u00B1 {std_ms:.4f} ms")
        return total_ms, avg_ms, std_ms

    # GPU timing
    elif device.lower() == "gpu":
        import cupy as cp
        for _ in range(n_warmup):
            run_kernel(name, device=device, **kwargs)
        cp.cuda.Stream.null.synchronize()

        for _ in range(n_warmup):
            start = cp.cuda.Event()
            end   = cp.cuda.Event()
            start.record()
            run_kernel(name, device=device, **kwargs)
            end.record()
            end.synchronize()

        run_times = []
        for _ in range(n_repeat):
            start = cp.cuda.Event()
            end   = cp.cuda.Event()
            start.record()
            run_kernel(name, device=device, **kwargs)
            end.record()
            end.synchronize()
            elapsed_time = cp.cuda.get_elapsed_time(start, end)  # in ms
            run_times.append(elapsed_time)
        total_ms = sum(run_times)
        avg_ms = total_ms / n_repeat
        std_ms = np.std(run_times)
        print(f"GPU: Total {n_repeat} runs: {total_ms:.2f} ms")
        print(f"GPU: Avg per run: {avg_ms:.2f} \u00B1 {std_ms:.4f} ms")
        return total_ms, avg_ms, std_ms
