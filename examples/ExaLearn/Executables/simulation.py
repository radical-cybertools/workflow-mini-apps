#!/usr/bin/env python

import os, sys, socket
import time
import argparse
import kernal as wf

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

    msz = args.mat_size

    for mi in range(rank, args.num_mult, size):
        elap = time.time()
        for ini in range(args.sim_inner_iter):
            wf.generateRandomNumber("cpu", msz * msz)
            wf.generateRandomNumber("cpu", msz * msz)
            wf.matMulGeneral("cpu", [msz, msz], [msz, msz], ([1], [0]))
        elap = time.time() - elap
        print("Rank is {}, mi is {}, takes {} second".format(rank, mi, elap))

        wf.writeNonMPI(4 * msz * msz)
        wf.writeNonMPI(4 * msz * msz)
   
    wf.writeNonMPI(args.write_size)
    wf.readNonMPI(args.read_size)
    end_time = time.time()
    print("Rank is {}, total running time is {} seconds".format(rank, end_time - start_time))

if __name__ == '__main__':
    main()

