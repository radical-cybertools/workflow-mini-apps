"""Compatibility module: ``import wfMiniAPI.kernel`` keeps working.

The kernels now live in ``wfMiniAPI.kernels``.  Existing workflows and task
scripts import this module (e.g. DeepDriveSim's ``import wfMiniAPI.kernel as wf``),
so it re-exports every name it exposed before the split.
"""

import os  # noqa: F401  (exposed before the split)
import sys  # noqa: F401
import time  # noqa: F401

import numpy as np  # noqa: F401

from .kernels import *  # noqa: F403
from .kernels._backends import *  # noqa: F403
from .kernels.compute import _axpy_fuse_fast  # noqa: F401
from .kernels import (  # noqa: F401
    MPIallGather,
    MPIallReduce,
    annotate_kernel,
    axpy,
    axpy_fast,
    dataCopyD2H,
    dataCopyH2D,
    fft,
    fftn,
    generateRandomNumber,
    get_device_module,
    inplaceCompute,
    kernel_params,
    list_kernels,
    matMulGeneral,
    matMulSimple2D,
    readNonMPI,
    readWithMPI,
    run_kernel,
    scatterAdd,
    sleep,
    time_kernel,
    top_k,
    writeNonMPI,
    writeSingleRank,
    writeWithMPI,
)
