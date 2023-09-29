#!/usr/bin/env python

import numpy as np
import cupy as cp
import io, os, sys, socket
import time
import argparse
import h5py

def parse_args():
    parser = argparse.ArgumentParser(description='Exalearn_miniapp_training')
    parser.add_argument('--num_epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--device', default='gpu',
                        help='Wheter this is running on cpu or gpu')
    parser.add_argument('--phase', type=int, default=0,
                        help='the current phase of workflow, phase0 will not read model')
    parser.add_argument('--data_root_dir', default='./',
                        help='the root dir of gsas output data')
    parser.add_argument('--model_dir', default='./',
                        help='the directory where save and load model')
    parser.add_argument('--num_sample', type=int, default=100,
                        help='num of samples in matrix mult')
    parser.add_argument('--num_mult', type=int, default=10,
                        help='number of matrix mult to perform')
    parser.add_argument('--dense_dim_in', type=int, default=12544,
                        help='dim for most heavy dense layer, input')
    parser.add_argument('--dense_dim_out', type=int, default=128,
                        help='dim for most heavy dense layer, output')
    parser.add_argument('--mat_size', type=int, default=5000,
                        help='the matrix with have size of mat_size * mat_size, should be the same as it is in simulation')
    parser.add_argument('--preprocess_time', type=float, default=20.0,
                        help='time for doing preprocess')
    parser.add_argument('--read_size', type=int, default=0,
                        help='size of bytes read from disk')
    parser.add_argument('--write_size', type=int, default=3500000,
                        help='size of bytes written to disk, -1 means write data to disk once')


    args = parser.parse_args()

    return args

def read_tmp_data(args):
    root_path = args.data_root_dir + '/phase{}'.format(args.phase) + '/'
    msz = args.mat_size
    if args.read_size == -1:
        read_time = 1
    else:
        read_time = int(args.read_size // (msz * 8))
    print("read_time = ", read_time)

    fname = root_path + 'all_tmp_data_0.hdf5'
    for i in range(read_time):
        with h5py.File(fname, 'r') as f:
            dataset_len = len(f.keys())
            D = f['tmp_{}'.format(i % dataset_len)][:]

def write_tmp_data(args):
    root_path = args.data_root_dir + '/phase{}'.format(args.phase) + '/'
    msz = args.mat_size
    if args.write_size == -1:
        write_time = 1
    else:
        write_time = int(args.write_size // (msz * 8))
    print("write_time = ", write_time)

    fname = root_path + 'all_tmp_data.hdf5'
    D = np.random.rand(msz)
    with h5py.File(fname, 'w') as f:
        for i in range(write_time):
            f.create_dataset("tmp_{}".format(i), data = D)

def preprocess(seconds):
    time.sleep(seconds)


def main():

    print("Temp for Darshan, ml, PID = {}, hostname = {}".format(os.getpid(), socket.gethostname()))
    start_time = time.time()

    args = parse_args()
    print(args)
    if args.device == 'gpu':
        print("gpu id is {}".format(cp.cuda.runtime.getDeviceProperties(0)['uuid']))

    preprocess(args.preprocess_time)
    read_tmp_data(args)

    X = np.random.rand(args.num_sample, args.dense_dim_in)
    X = np.float32(X)
    w = np.random.rand(args.dense_dim_in, args.dense_dim_out)
    w = np.float32(w)
    w_d = cp.asarray(w)

    for epoch in range(args.num_epochs):
        tt = time.time()

        if args.device == 'cpu':
            for ii in range(args.train_inner_iter):
                for index, mi in enumerate(range(args.num_mult)):
                    R_temp = np.matmul(X[index], y[index])
            print("Epoch is {}, mult takes {}".format(epoch, time.time() - tt))
            tt = time.time()

        elif args.device == 'gpu':
            X_d = cp.asarray(X)
            print("epoch is {}, data movementi (CPU->GPU) takes {}".format(epoch, time.time() - tt))
            tt = time.time()
            for ii in range(args.num_mult):
                z_d = cp.dot(X_d, w_d)
                w_d = w_d + 0.1
            print("epoch is {}, mult takes {}".format(epoch, time.time() - tt))
            tt = time.time()

    w_temp = cp.asnumpy(w_d)
    write_tmp_data(args)

    end_time = time.time()
    print("Total running time is {}) seconds".format(end_time - start_time))

if __name__ == '__main__':
    main()
