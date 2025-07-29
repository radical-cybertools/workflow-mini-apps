#!/usr/bin/env python

import io, os, sys, socket
import time
import argparse
import wfMiniAPI.kernel as kernel

def parse_args():
    parser = argparse.ArgumentParser(description='Bayasian ML Training')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--num_sample', type=int, default=512,
                        help='num of samples in matrix mult')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size in training')
    parser.add_argument('--device', default='gpu',
                        help='Wheter this is running on cpu or gpu')
    parser.add_argument('--dense_dim_in', type=int, default=2048,
                        help='dim for most heavy dense layer, input')
    parser.add_argument('--dense_dim_out', type=int, default=512,
                        help='dim for most heavy dense layer, output')
    parser.add_argument('--log_freq', type=int, default=20,
                        help='epochs between logging outputs (default: 20)')
    
    # Tuning knobs
    parser.add_argument('--write_size', type=int, default=0,
                        help='size of bytes written to disk')
    parser.add_argument('--num_mult', type=int, default=10,
                        help='number of matrix mult to perform')

    # Task related parameters
    parser.add_argument('--experiment_dir', required=True,
                        help='the root dir of gsas output data')
    parser.add_argument('--task_index', required=True,
                        help='the index of task to prevent colliding')

    args = parser.parse_args()

    return args


def main():

    start_time = time.time()

    args = parse_args()
    print(args)

    root_path = args.experiment_dir + '/{}'.format(args.task_index) + '/'
    print("root_path for data = ", root_path)

    kernel.generateRandomNumber(args.device, args.dense_dim_in * args.dense_dim_out)
    if args.device == 'gpu':
        kernel.dataCopyH2D(args.dense_dim_in * args.dense_dim_out)

    for epoch in range(args.num_epochs):
        num_batch = args.num_sample // args.batch_size
        for _ in range(num_batch):
            kernel.dataCopyH2D(args.batch_size * args.dense_dim_in)
            for ii in range(args.num_mult):
                kernel.matMulGeneral(args.device, [args.batch_size, args.dense_dim_in], [args.dense_dim_in, args.dense_dim_out], ([1], [0]))
                kernel.axpy_fast(args.device, args.dense_dim_in * args.dense_dim_out)
        if epoch % args.log_freq == 0:
            dir_name = os.path.join(root_path, f"./epoch_{epoch}")
            os.makedirs(dir_name, exist_ok=True)
            kernel.writeSingleRank(args.write_size, dir_name)

    if args.device == 'gpu':
        kernel.dataCopyD2H(args.dense_dim_in * args.dense_dim_out)

    end_time = time.time()
    print("Total running time is {} seconds".format(end_time - start_time))

if __name__ == '__main__':
    main()
