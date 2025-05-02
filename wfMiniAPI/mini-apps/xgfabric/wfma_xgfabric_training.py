#!/usr/bin/env python3

import io, os, sys, socket
import time
import argparse

from wfMiniAPI import kernel as wf

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


def main():

    print("Temp for Darshan, ml, PID = {}, hostname = {}".format(os.getpid(), socket.gethostname()))
    start_time = time.time()

    args = parse_args()
    print(args)
    device = args.device
    if device == 'gpu':
        print("gpu id is {}".format(cp.cuda.runtime.getDeviceProperties(0)['uuid']))

    wf.readNonMPI(args.read_size)
    wf.sleep(args.preprocess_time)

    for epoch in range(args.num_epochs):
        tt = time.time()

        if device == 'cpu':
            for ii in range(args.train_inner_iter):
                for index, mi in enumerate(range(0, args.num_mult, args.sim_rank)):
                    wf.matMulGeneral(device, [args.mat_size, args.mat_size], [args.mat_size, args.mat_size], ([1], [0]))
            print("epoch is {}, mult takes {}".format(epoch, time.time() - tt))
            tt = time.time()

        elif device == 'gpu':
            tile_time = (args.num_mult + args.sim_rank - 1) // args.sim_rank
            wf.dataCopyH2D(2 * args.mat_size * args.mat_size * tile_time)
            print("epoch is {}, data movementi (CPU->GPU) takes {}".format(epoch, time.time() - tt))
            tt = time.time()
            for ii in range(args.train_inner_iter):
                for index, mi in enumerate(range(0, args.num_mult, args.sim_rank)):
                    wf.matMulGeneral(device, [args.mat_size, args.mat_size], [args.mat_size, args.mat_size], ([1], [0]))
            print("epoch is {}, mult takes {}".format(epoch, time.time() - tt))
            tt = time.time()
            wf.dataCopyH2D(args.mat_size * args.mat_size * tile_time)
            print("epoch is {}, data movementi (GPU->CPU) takes {}".format(epoch, time.time() - tt))
            print(R_temp.shape)
            tt = time.time()

        for ii in range(args.num_allreduce):
            wf.allReduce("cpu", args.mat_size * args.mat_size)
        print("epoch is {}, allreduce takes {}".format(epoch, time.time() - tt))

    wf.writeSingleRank(4 * args.mat_size * args.mat_size)
    end_time = time.time()
    print("Total running time is {}) seconds".format(end_time - start_time))

if __name__ == '__main__':
    main()
