#!/usr/bin/env python3

import numpy as np
print(np.__version__)

import cupy as cp
print(cp.__version__)

import numba
print(numba.__version__)

print(cp.cuda.runtime.getDeviceProperties(0)['uuid'])
print(cp.cuda.runtime.getDeviceProperties(1)['uuid'])
print(cp.cuda.runtime.getDeviceProperties(2)['uuid'])
print(cp.cuda.runtime.getDeviceProperties(3)['uuid'])

a_d = cp.arange(100, dtype=cp.float32)
a_h = cp.asnumpy(a_d)
b_h = np.arange(100, dtype=np.float32)
b_d = cp.asarray(b_h)
print(a_h)
print(a_d)
print(b_h)
print(b_d)


