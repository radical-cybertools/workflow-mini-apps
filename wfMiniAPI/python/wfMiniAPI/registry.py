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
