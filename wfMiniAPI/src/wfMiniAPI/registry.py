import inspect
from collections import OrderedDict

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

def time_kernel(name, device, n_warmup=1, n_repeat=20, **kwargs):
    xp = get_device_module(device)

    # CPU timing
    if device.lower() == "cpu":
        for _ in range(n_warmup):
            run_kernel(name, device=device, **kwargs)
        t0 = time.time()
        for _ in range(n_repeat):
            run_kernel(name, device=device, **kwargs)
        t1 = time.time()

        total_ms = (t1 - t0) * 1000
        avg_ms   = total_ms / n_repeat
        print(f"CPU: Total {n_repeat} runs: {total_ms:.2f} ms")
        print(f"CPU: Avg per run: {avg_ms:.2f} ms")
        return total_ms, avg_ms

    # GPU timing
    elif device.lower() == "gpu":
        # warmup + sync
        for _ in range(n_warmup):
            run_kernel(name, device=device, **kwargs)
        cp.cuda.Stream.null.synchronize()

        start = cp.cuda.Event()
        end   = cp.cuda.Event()
        start.record()
        for _ in range(n_repeat):
            run_kernel(name, device=device, **kwargs)
        end.record()
        end.synchronize()

        total_ms = cp.cuda.get_elapsed_time(start, end)
        avg_ms   = total_ms / n_repeat
        print(f"GPU: Total {n_repeat} runs: {total_ms:.2f} ms")
        print(f"GPU: Avg per run: {avg_ms:.2f} ms")
        return total_ms, avg_ms
