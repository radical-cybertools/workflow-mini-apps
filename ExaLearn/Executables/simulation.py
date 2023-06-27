#!/usr/bin/env python

import os,sys
import time
import numpy as np
import argparse
from mpi4py import MPI

def parse_args():
    parser = argparse.ArgumentParser(description='Exalearn_miniapp_simulation')
    parser.add_argument('--phase', type=int, default=0,
                        help='the current phase of workflow, in miniapp all phases do the same thing except rng')
    parser.add_argument('--mat_size', type=int, default=3000,
                    help='the matrix with have size of mat_size * mat_size')
    parser.add_argument('--data_root_dir', default='./',
                    help='the root dir of gsas output data')
    parser.add_argument('--num_mult', type=int, default=10,
                    help='number of matrix mult to perform, need to be larger than num_worker!')
    parser.add_argument('--inner_iter', type=int, default=10,
                    help='number of inner iter for each matrix mult. Used to control sim workload size')
   
    args = parser.parse_args()

    return args

def matMult(a, b , out):
    out = np.matmul(a,b)

def main():

    start_time = time.time()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    args = parse_args()
    if rank == 0:
        print(args)
    if args.num_mult < size:
        sys.exit("Error: num_mult can not be smaller than num_worker!")

    root_path = args.data_root_dir + '/phase{}'.format(args.phase) + '/'
    if rank == 0:
        print("root_path for data = ", root_path)

    seed = 27 + rank * 100 + args.phase     #Make sure different running has different seed
    np.random.seed(seed)  

    msz = args.mat_size

    print("Rank is ", rank, " size is ", size, " seed is ", seed);

    filename_X = root_path + 'all_X_data_rank_{}.npy'.format(rank)
    os.makedirs(os.path.dirname(filename_X), exist_ok=True)
    filename_Y = root_path + 'all_Y_data_rank_{}.npy'.format(rank)
    os.makedirs(os.path.dirname(filename_Y), exist_ok=True)

    for mi in range(rank, args.num_mult, size):
        elap = time.time()
#        print("A = ", A)
#        print("B = ", B)
        for ini in range(args.inner_iter):
            A = np.random.rand(msz,msz)
            B = np.random.rand(msz,msz)
            C = np.matmul(A,B)
#        print("C = ", C)
        elap = time.time() - elap
        print("Rank is {}, mi is {}, takes {} second".format(rank, mi, elap))

        with open(filename_X, 'wb') as f:
            np.save(f, C)
        with open(filename_Y, 'wb') as f:
            np.save(f, C)

    end_time = time.time()
    print("Rank is {}, total running time is {}) seconds".format(rank, end_time - start_time))

if __name__ == '__main__':
    main()

