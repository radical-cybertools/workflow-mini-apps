#!/usr/bin/env python3

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

a = np.arange(10*rank, 10*(rank+1))
b = np.zeros_like(a)
print("rank = {}".format(rank), " array = ", a)

comm.Allreduce(a, b, op=MPI.SUM)
print("rank = {}".format(rank), " final = ", b)
