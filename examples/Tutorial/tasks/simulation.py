#!/usr/bin/env python

import io, os, sys, socket
import time
import argparse
import wfMiniAPI.kernel as kernel

def parse_args():

    parser = argparse.ArgumentParser(description="Molecular Dynamics Simulation Configuration")

    # Simulation parameters
    parser.add_argument('--N_atoms', type=int, default=10000,
                        help='Number of particles (default: 10000)')
    parser.add_argument('--grid_size', type=int, default=64,
                        help='PME grid size (default: 64)')
    parser.add_argument('--neighbor_freq', type=int, default=10,
                        help='Steps between neighboring list updates (default: 10)')
    parser.add_argument('--log_freq', type=int, default=20,
                        help='Steps between logging outputs (default: 20)')
    parser.add_argument('--n_steps', type=int, default=100,
                        help='Total MD steps to emulate (default: 100)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'],
                        help='Device to run the simulation on (default: cpu)')

    # Tuning knobs
    parser.add_argument('--n_force', type=int, default=100,
                        help='Short-range AXPY (default: 100)')
    parser.add_argument('--n_fft', type=int, default=2,
                        help='Number of forward+inverse FFTs (default: 2)')
    parser.add_argument('--n_int', type=int, default=20,
                        help='Integration AXPY count (default: 20)')
    parser.add_argument('--bytes_per_atom', type=int, default=144,
                        help='Number of bytes per atom (default: 144)')

    # Task related parameters
    parser.add_argument('--experiment_dir', required=True,
                        help='the root dir of gsas output data')
    parser.add_argument('--task_index', required=True,
                        help='the index of task to prevent colliding')

    args = parser.parse_args()
    args.io_bytes = args.N_atoms * args.bytes_per_atom 

    return args

def main():

    start_time = time.time()

    args = parse_args()
    print(args)

    root_path = args.experiment_dir + '/{}'.format(args.task_index) + '/'
    print("root_path for data = ", root_path)

    kernel.generateRandomNumber(args.device, args.N_atoms * 3 * 3)
    print("After Initialization takes ", time.time() - start_time)

    for i in range(args.n_steps):
        if i % args.neighbor_freq == 0:
            kernel.matMulGeneral(args.device, size_a=(args.N_atoms,3), size_b=(3, args.N_atoms), axis=1)
        for j in range(args.n_force):
            kernel.axpy_fast(args.device, 3*args.N_atoms)
        for j in range(args.n_fft):
            kernel.fftn(args.device, (args.grid_size, args.grid_size, args.grid_size), 'complexF', (0,1,2))
        for j in range(args.n_int):
            kernel.axpy_fast(args.device, 3*args.N_atoms)
        if i % log_freq == 0:
            dir_name = os.path.join(root_path, f"./log_step_{i}")
            os.makedirs(dir_name, exist_ok=True)
            kernel.writeSingleRank(io_bytes, dir_name)
    print("After main loop takes ", time.time() - start_time)

    if args.device == 'gpu':
        kernel.dataCopyD2H(args.N_atoms * 3 * 3)
    print("Total running time is {} seconds".format(end_time - start_time))

if __name__ == '__main__':
    main()

