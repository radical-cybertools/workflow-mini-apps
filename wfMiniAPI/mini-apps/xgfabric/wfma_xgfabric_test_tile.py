#!/usr/bin/env python3

import numpy as np

a = np.array([[1,2],[3,4]])
print(a)
b = np.tile(a, (3,1,1))
print(b.shape)
print(b[0,:])
print(b[1,:])
print(b[2,:])
