#!/usr/bin/env python

import os, sys, socket
import time
import argparse
import kernel as wf

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
    parser.add_argument('--instance_index', type=int, required=True,
                        help='use to distinguish different sim task. Should be from 0~n-1')

    args = parser.parse_args()

    return args

def main():

    print("Temp for Darshan, sim, PID = {}, hostname = {}".format(os.getpid(), socket.gethostname()))
    start_time = time.time()

    args = parse_args()
    print(args)

    root_path = args.data_root_dir + '/phase{}'.format(args.phase) + '/'
    print("root_path for data = ", root_path)

    msz = args.mat_size
    device = "gpu"

    wf.generateRandomNumber(device, msz)
    wf.generateRandomNumber(device, msz)

    print(time.time() - start_time)
    for mi in range(args.num_step):
        elap = time.time()
        wf.axpy(device, msz)
        wf.axpy(device, msz)
        wf.generateRandomNumber(device, msz * msz)
        wf.implaceCompute(device, msz * msz, 1)
        wf.matMulGeneral(device, [msz, msz], [msz], ([1], [0]))
    print(time.time() - start_time)

    wf.dataCopyD2H(msz)
    wf.dataCopyD2H(msz)
    print(time.time() - start_time)

    wf.writeNonMPI(args.write_size, root_path, args.instance_index)
    wf.readNonMPI(args.read_size, root_path, args.instance_index)

    end_time = time.time()
    print("Total running time is {} seconds".format(end_time - start_time))

if __name__ == '__main__':
    main()

