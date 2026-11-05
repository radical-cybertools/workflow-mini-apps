"""Miscellaneous kernels and device selection."""

import time

import numpy as np

from ._backends import *  # noqa: F403 -- cp, nccl, MPI, h5py exist only when installed
from .registry import annotate_kernel


@annotate_kernel
def sleep(seconds):
    time.sleep(seconds)

@annotate_kernel
def get_device_module(device):
    if device == "gpu":
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not installed. Install CuPy to use GPU capabilities.")
        return cp
    else:
        return np
