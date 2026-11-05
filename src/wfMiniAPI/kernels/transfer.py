"""Host/device data movement kernels."""

import numpy as np

from ._backends import *  # noqa: F403 -- cp, nccl, MPI, h5py exist only when installed
from .registry import annotate_kernel


@annotate_kernel
def dataCopyH2D(data_size):
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not installed. Install CuPy to use GPU capabilities.")
    else:
        data_h = np.empty(data_size, dtype=np.float32)
        data_d = cp.asarray(data_h)

@annotate_kernel
def dataCopyD2H(data_size):
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not installed. Install CuPy to use GPU capabilities.")
    else:
        data_d = cp.empty(data_size, dtype=cp.float32)
        data_h = cp.asnumpy(data_d)
