#!/usr/bin/env python

import numpy as np
import cupy as cp
import io, os, sys, socket
import time
import argparse
import h5py
from mpi4py import MPI

def parse_args():
    parser = argparse.ArgumentParser(description='Exalearn_miniapp_training')
    parser.add_argument('--num_epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--device', default='cpu',
                        help='Wheter this is running on cpu or gpu')
    parser.add_argument('--phase', type=int, default=0,
                        help='the current phase of workflow, phase0 will not read model')
    parser.add_argument('--data_root_dir', default='./',
                        help='the root dir of gsas output data')
    parser.add_argument('--model_dir', default='./',
                        help='the directory where save and load model')
    parser.add_argument('--mat_size', type=int, default=3000,
                        help='the matrix with have size of mat_size * mat_size')
    parser.add_argument('--num_mult', type=int, default=10,
                        help='number of matrix mult to perform, need to be larger than num_worker!')
    parser.add_argument('--sim_rank', type=int, default=1,
                        help='number of rank used for simulation. This is needed to determine the size of data in those files')
    parser.add_argument('--preprocess_time', type=float, default=5.0,
                        help='time for doing preprocess')
    parser.add_argument('--train_inner_iter', type=int, default=1,
                        help='inner iteration of mult and allreduce')
    parser.add_argument('--num_allreduce', type=int, default=1,
                        help='the number of allreduce op performed')
    parser.add_argument('--read_size', type=int, default=0,
                        help='size of bytes read from disk')

    args = parser.parse_args()

    return args

def read_tmp_data(args, rank, size):
    root_path = args.data_root_dir + '/phase{}'.format(args.phase) + '/'
    msz = args.mat_size
    read_time = int(args.read_size // (msz // 4 * msz // 4 * 8 * size))
    print("read_time = ", read_time)
    fname = root_path + 'all_tmp_data_rank_{}.hdf5'.format(rank)
    for i in range(read_time):
        with h5py.File(fname, 'r') as f:
            dataset_len = len(f.keys())
            print("dataset_len = ", dataset_len)
            D = f['tmp_{}'.format(i % dataset_len)][:]

def load_data(args, rank):
    root_path = args.data_root_dir + '/phase{}'.format(args.phase) + '/'
    msz = args.mat_size
    X_scaled = np.load(root_path + 'all_X_data_rank_0.npy')
    y_scaled = np.load(root_path + 'all_Y_data_rank_0.npy')
    tile_time = (args.num_mult + args.sim_rank - 1 - rank) // args.sim_rank
    X = np.tile(X_scaled, (tile_time, 1, 1))
    y = np.tile(y_scaled, (tile_time, 1, 1))
    print(X.shape, y.shape)
    return X, y

def preprocess(seconds):
    time.sleep(seconds)


def main():

    print("Temp for Darshan, ml, PID = {}, hostname = {}".format(os.getpid(), socket.gethostname()))
    start_time = time.time()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    args = parse_args()
    if rank == 0:
        print(args)
    if args.device == 'gpu':
        print("Rank is {}, gpu id is {}".format(rank, cp.cuda.runtime.getDeviceProperties(0)['uuid']))

    read_tmp_data(args, rank, size)
    X, y = load_data(args, rank)
    X = np.float32(X)
    y = np.float32(y)
    preprocess(args.preprocess_time)
    for epoch in range(args.num_epochs):
        tt = time.time()

        if args.device == 'cpu':
            for ii in range(args.train_inner_iter):
                for index, mi in enumerate(range(rank, args.num_mult, args.sim_rank)):
                    R_temp = np.matmul(X[index], y[index])
            print("Rank is {}, epoch is {}, mult takes {}".format(rank, epoch, time.time() - tt))
            tt = time.time()

        elif args.device == 'gpu':
            X_d = cp.asarray(X)
            y_d = cp.asarray(y)
            print("Rank is {}, epoch is {}, data movementi (CPU->GPU) takes {}".format(rank, epoch, time.time() - tt))
            print(X_d.shape, y_d.shape)
            tt = time.time()
            for ii in range(args.train_inner_iter):
                for index, mi in enumerate(range(rank, args.num_mult, args.sim_rank)):
                    R_temp_d = cp.dot(X_d[index], y_d[index])
            print("Rank is {}, epoch is {}, mult takes {}".format(rank, epoch, time.time() - tt))
            tt = time.time()
            X_temp = cp.asnumpy(X_d)
            y_temp = cp.asnumpy(y_d)
            R_temp = cp.asnumpy(R_temp_d)
            print("Rank is {}, epoch is {}, data movementi (GPU->CPU) takes {}".format(rank, epoch, time.time() - tt))
            print(R_temp.shape)
            tt = time.time()

        for ii in range(args.num_allreduce):
            R = np.zeros_like(R_temp)
            comm.Allreduce(R_temp, R, op=MPI.SUM)
        print("Rank is {}, epoch is {}, allreduce takes {}".format(rank, epoch, time.time() - tt))

    if rank == 0:
        with open(args.model_dir + '/result_phase{}.npy'.format(args.phase), 'wb') as f:
            np.save(f, R)

    end_time = time.time()
    print("Rank is {}, total running time is {}) seconds".format(rank, end_time - start_time))

if __name__ == '__main__':
    main()
