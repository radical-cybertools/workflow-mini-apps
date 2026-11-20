"""Computation kernels."""

import numpy as np

from ._backends import *  # noqa: F403 -- cp, nccl, MPI, h5py exist only when installed
from .misc import get_device_module
from .registry import annotate_kernel


@annotate_kernel
def matMulSimple2D(device, size):
    xp = get_device_module(device)
    matrix_a = xp.empty((size, size), dtype=xp.float32)
    matrix_b = xp.empty((size, size), dtype=xp.float32)
    matrix_c = xp.matmul(matrix_a, matrix_b)

@annotate_kernel
def matMulGeneral(device, size_a, size_b, axis):
    xp = get_device_module(device)
    matrix_a = xp.empty(tuple(size_a), dtype=xp.float32)
    matrix_b = xp.empty(tuple(size_b), dtype=xp.float32)
    matrix_c = xp.tensordot(matrix_a, matrix_b, axis)

@annotate_kernel
def fft(device, data_size, type_in, transform_dim):
    xp = get_device_module(device)
    if type_in == "float":
        data_in = xp.empty(tuple(data_size), dtype=xp.float32)
    elif type_in == "double":
        data_in = xp.empty(tuple(data_size), dtype=xp.float64)
    elif type_in == "complexF":
        data_in = xp.empty(tuple(data_size), dtype=xp.complex64)
    elif type_in == "complexD":
        data_in = xp.empty(tuple(data_size), dtype=xp.complex128)
    else:
        raise TypeError("In fft call, type_in must be one of the following: [float, double, complexF, complexD]")

    out = xp.fft.fft(data_in, axis=transform_dim)

@annotate_kernel
def fftn(device, data_size, type_in, transform_dim):
    xp = get_device_module(device)
    if type_in == "float":
        data_in = xp.empty(tuple(data_size), dtype=xp.float32)
    elif type_in == "double":
        data_in = xp.empty(tuple(data_size), dtype=xp.float64)
    elif type_in == "complexF":
        data_in = xp.empty(tuple(data_size), dtype=xp.complex64)
    elif type_in == "complexD":
        data_in = xp.empty(tuple(data_size), dtype=xp.complex128)
    else:
        raise TypeError("In fftn call, type_in must be one of the following: [float, double, complexF, complexD]")

    out = xp.fft.fftn(data_in, axes=transform_dim)

@annotate_kernel
def axpy(device, size):
    xp = get_device_module(device)
    x = xp.empty(size, dtype=xp.float32)
    y = xp.empty(size, dtype=xp.float32)
    y += 1.01 * x

#_axpy_fuse = cp.ElementwiseKernel(
#    'float32 alpha, raw float32 x',
#    'raw float32 y',               
#    'y[i] += alpha * x[i]',        
#    'axpy_fuse_kernel',
#    no_return=True                 
#)
#
#@annotate_kernel
#def axpy(device, size):
#    xp = get_device_module(device)
#    x = xp.empty(size, dtype=xp.float32)
#    y = xp.empty(size, dtype=xp.float32)
#    if xp == np:
#        y += 1.01 * x
#    elif xp == cp:
#        _axpy_fuse(1.01, x, y, size=size)

_axpy_fuse_fast = cp.ElementwiseKernel(
    'float32 alpha, raw float32 x',
    'raw float32 y',
    'y[i] += alpha * x[i]',
    'axpy_fuse_kernel',
    no_return=True
) if CUPY_AVAILABLE else None

@annotate_kernel
def axpy_fast(device, size):
    xp = get_device_module(device)
    x = xp.empty(size, dtype=xp.float32)
    y = xp.empty(size, dtype=xp.float32)
    if xp == np:
        y += 1.01 * x
    elif xp == cp:
        _axpy_fuse_fast(1.01, x, y, size=size)

@annotate_kernel
def inplaceCompute(device, size, num_op, op):
    xp = get_device_module(device)
    x = xp.empty(size, dtype=xp.float32)
    if isinstance(op, str):
        try:
            func = getattr(xp, op)
        except AttributeError:
            raise ValueError(f"The operator '{op}' is not available in the module {xp.__name__}.")
    elif callable(op):
        func = op
    else:
        raise ValueError("Operator must be either a string or a callable function.")
    for _ in range(num_op):
        x = func(x)

@annotate_kernel
def generateRandomNumber(device, size):
    xp = get_device_module(device)
    x = xp.random.rand(size)

@annotate_kernel
def scatterAdd(device, x_size, y_size):
    xp = get_device_module(device)
    y = xp.empty(y_size, dtype=xp.float32)
    x = xp.empty(x_size, dtype=xp.float32)
    idx = xp.random.randint(0, x_size, y_size)
    if xp == np:
        y += x[idx]
    elif xp == cp:
        cp.add.at(y, idx, x)

@annotate_kernel
def top_k(device, size, k):
    xp = get_device_module(device)
    arr = xp.empty(size, dtype=xp.float32)
    indices = xp.argpartition(-arr, k)[:k]
    sorted_indices = indices[xp.argsort(-arr[indices])]
    top_values = arr[sorted_indices]
