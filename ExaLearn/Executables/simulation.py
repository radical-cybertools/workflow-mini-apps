#!/usr/bin/env python

import os, sys, socket
import time
import numpy as np
import argparse
import h5py
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
    parser.add_argument('--sim_inner_iter', type=int, default=10,
                    help='number of inner iter for each matrix mult. Used to control sim workload size')
    parser.add_argument('--write_size', type=int, default=-1,
                    help='size of bytes written to disk, -1 means write data to disk once')
    parser.add_argument('--read_size', type=int, default=0,
                    help='size of bytes read from disk')

    args = parser.parse_args()

    return args

def matMult(a, b , out):
    out = np.matmul(a,b)

def main():

    print("Temp for Darshan, sim, PID = {}, hostname = {}".format(os.getpid(), socket.gethostname()))
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

    if args.write_size == -1:
        write_time = 1
    else:
        print(args.write_size, msz // 4, size, args.write_size // (msz // 4 * msz // 4 * 8 * size))
        write_time = int(args.write_size // (msz // 4 * msz // 4 * 8 * size))
    print("write_time = {}".format(write_time))
    
    if args.read_size == -1:
        read_time = 1
    else:
        read_time = int(args.read_size // (msz // 4 * msz // 4 * 8 * size))
    print("read_time = {}".format(read_time))

    for mi in range(rank, args.num_mult, size):
        elap = time.time()
#        print("A = ", A)
#        print("B = ", B)
        for ini in range(args.sim_inner_iter):
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
    
    fname = root_path + 'all_tmp_data_rank_{}.hdf5'.format(rank)
    D = np.random.rand(msz//4,msz//4)
    with h5py.File(fname, 'w') as f:
        for i in range(write_time):
            f.create_dataset("tmp_{}".format(i), data = D)
    for i in range(read_time):
        fname = root_path + 'all_tmp_data_rank_{}.hdf5'.format(rank)
        with h5py.File(fname, 'r') as f:
            D = f['tmp_{}'.format(i % write_time)][:]

    end_time = time.time()
    print("Rank is {}, total running time is {} seconds".format(rank, end_time - start_time))

if __name__ == '__main__':
    main()

