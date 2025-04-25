#!/usr/bin/env python

import os, sys, socket
import time
import argparse
import kernel as wf
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
    parser.add_argument('--input_file', type=str, default=None,
                    help='path to an input file to read using the readFile kernel')
    parser.add_argument('--read_ratio', type=float, default=1.0,
                    help='ratio of file to read (0.0-1.0)')
    parser.add_argument('--scale_matrix', action='store_true',
                    help='scale matrix size based on input file size')

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
    
    # Default matrix size
    msz = args.mat_size
    
    # Read input file if provided and consider scaling matrix size
    bytes_read = 0
    reference_size = 50 * 1024 * 1024  # 50MB as reference file size
    
    # setting output file name
    output_file = os.path.join(root_path, f"output_{rank}.txt")

    if args.input_file:
        if rank == 0:
            print(f"Reading input file: {args.input_file} with ratio: {args.read_ratio}")
        bytes_read = wf.readFile(args.input_file, args.read_ratio)
        print(f"Rank {rank} read {bytes_read} bytes from input file")
        
        # Scale matrix size based on input file size if requested
        if args.scale_matrix and bytes_read > 0:
            # Calculate scaling factor (square root relationship with file size)
            scale_factor = (bytes_read / reference_size) ** 0.5
            new_msz = int(msz * scale_factor)
            # Ensure new size is at least 100
            new_msz = max(100, new_msz)
            
            if rank == 0:
                print(f"Scaling matrix size: original={msz}, new={new_msz} (scale factor: {scale_factor:.2f})")
            msz = new_msz
        # set the output file name based on input file
        output_file =f"{args.input_file}_output.txt"

    for mi in range(rank, args.num_mult, size):
        elap = time.time()
        for ini in range(args.sim_inner_iter):
            wf.generateRandomNumber("cpu", msz * msz)
            wf.generateRandomNumber("cpu", msz * msz)
            wf.matMulGeneral("cpu", [msz, msz], [msz, msz], ([1], [0]))
        elap = time.time() - elap
        print("Rank is {}, mi is {}, takes {} second".format(rank, mi, elap))

        wf.writeFile(output_file, args.write_size)
        wf.writeFile(output_file, args.write_size)
   
    wf.writeFile(output_file, args.write_size)
    if not args.input_file:
        wf.readNonMPI(args.read_size)
    end_time = time.time()
    print("Rank is {}, total running time is {} seconds".format(rank, end_time - start_time))

if __name__ == '__main__':
    main()

