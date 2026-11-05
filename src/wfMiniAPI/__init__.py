"""wfMiniAPI: emulation kernels for workflow mini-apps.

Importing the package has no side effects.  Submodules load on first use, so
``import wfMiniAPI; wfMiniAPI.kernel`` keeps working as before.
"""

import importlib

__all__ = ["kernel", "registry", "sim"]

__version__ = "0.2.0.dev0"

_LAZY = {"kernel": ".kernel", "kernels": ".kernels", "registry": ".kernels.registry", "sim": ".sim"}


def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name], __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
