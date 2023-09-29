#!/usr/bin/env python

import os, sys, socket
import time
import numpy as np
import cupy as cp
import argparse
import h5py

def parse_args():
    parser = argparse.ArgumentParser(description='Exalearn_miniapp_simulation')
    parser.add_argument('--phase', type=int, default=0,
                        help='the current phase of workflow, in miniapp all phases do the same thing except rng')
    parser.add_argument('--mat_size', type=int, default=5000,
                        help='the matrix with have size of mat_size * mat_size')
    parser.add_argument('--data_root_dir', default='./',
                        help='the root dir of gsas output data')
    parser.add_argument('--num_step', type=int, default=10000,
                        help='number of matrix mult to perform, need to be larger than num_worker!')
    parser.add_argument('--write_size', type=int, default=3500000,
                        help='size of bytes written to disk, -1 means write data to disk once')
    parser.add_argument('--read_size', type=int, default=6000000,
                        help='size of bytes read from disk')

    args = parser.parse_args()

    return args

def main():

    print("Temp for Darshan, sim, PID = {}, hostname = {}".format(os.getpid(), socket.gethostname()))
    start_time = time.time()

    args = parse_args()
    print(args)

    root_path = args.data_root_dir + '/phase{}'.format(args.phase) + '/'
    print("root_path for data = ", root_path)

    seed = 27 + os.getpid() * 100 + args.phase     #Make sure different running has different seed
    cp.random.seed(seed)  

    msz = args.mat_size

    filename_X = root_path + 'all_X_data.npy'
    os.makedirs(os.path.dirname(filename_X), exist_ok=True)

    if args.write_size == -1:
        write_time = 0
    else:
        write_time = int(args.write_size // (msz * 8))
    print("num_write = {}".format(write_time))
    
    if args.read_size == -1:
        read_time = 1
    else:
        read_time = int(args.read_size // (msz * 8))
    print("num_read = {}".format(read_time))

    va_d = cp.random.rand(msz)
    vb_d = cp.random.rand(msz)

    print(time.time() - start_time)
    for mi in range(args.num_step):
        elap = time.time()
        va_d = vb_d * 0.1 + va_d
        vb_d = va_d * 0.1 + vb_d
        mat_d = cp.random.rand(msz, msz)
        mat_d = cp.sin(mat_d)
        va_d = cp.dot(mat_d, vb_d)
    print(time.time() - start_time)

    va = cp.asnumpy(va_d)
    vb = cp.asnumpy(vb_d)
    print(time.time() - start_time)

    fname = root_path + 'all_tmp_data.hdf5'
    D = np.random.rand(msz)
    with h5py.File(fname, 'w') as f:
        for i in range(write_time):
            f.create_dataset("tmp_{}".format(i), data = D)
    for i in range(read_time):
        fname = root_path + 'all_tmp_data.hdf5'
        with h5py.File(fname, 'r') as f:
            D = f['tmp_{}'.format(i % write_time)][:]

    end_time = time.time()
    print("Total running time is {} seconds".format(end_time - start_time))

if __name__ == '__main__':
    main()

