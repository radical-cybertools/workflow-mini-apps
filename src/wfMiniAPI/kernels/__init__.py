"""wfMiniAPI emulation kernels.

Modules are imported in the order the kernels were defined in the former
``wfMiniAPI/kernel.py``, so ``list_kernels()`` keeps its order.
"""

from .registry import annotate_kernel, kernel_params, list_kernels, run_kernel, time_kernel
from ._backends import *  # noqa: F403
from .misc import get_device_module, sleep
from .io import readNonMPI, readWithMPI, writeNonMPI, writeSingleRank, writeWithMPI
from .comm import MPIallGather, MPIallReduce
from .transfer import dataCopyD2H, dataCopyH2D
from .compute import (
    axpy,
    axpy_fast,
    fft,
    fftn,
    generateRandomNumber,
    inplaceCompute,
    matMulGeneral,
    matMulSimple2D,
    scatterAdd,
    top_k,
)

__all__ = [
    "annotate_kernel", "kernel_params", "list_kernels", "run_kernel", "time_kernel",
    "get_device_module", "sleep",
    "readNonMPI", "readWithMPI", "writeNonMPI", "writeSingleRank", "writeWithMPI",
    "MPIallGather", "MPIallReduce",
    "dataCopyD2H", "dataCopyH2D",
    "axpy", "axpy_fast", "fft", "fftn", "generateRandomNumber", "inplaceCompute",
    "matMulGeneral", "matMulSimple2D", "scatterAdd", "top_k",
]
