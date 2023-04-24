#!/usr/bin/env python

import os,sys
import threading
import time
import numpy as np
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='Exalearn_miniapp_simulation')
    parser.add_argument('--phase', type=int, default=0,
                    help='the current phase of workflow, phase0 will not read model')
    parser.add_argument('--mat_size', type=int, default=3000,
                    help='the matrix with have size of mat_size * mat_size')
    parser.add_argument('--data_root_dir', default='./',
                    help='the root dir of gsas output data')
    parser.add_argument('--exec_pattern', default='single-thread',
                    help='running pattern, can be single-thread, multi-thread or MPI')
#FIXME default should ne total CPU -1
    parser.add_argument('--num_CPU', type=int, default=9,
                    help='number of CPU to use for multi-thread if not specified it will use 3x3')
    args = parser.parse_args()

    if args.exec_pattern == "multi-thread":
        args.inner_bsz = int(np.sqrt(args.num_CPU))

    return args


def matMult(a, b , out):
    out = np.matmul(a,b)

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (nrows, ncols, n, m) where
    n * nrows, m * ncols = arr.shape.
    This should be a view of the original array.
    """
    h, w = arr.shape
    n, m = h // nrows, w // ncols
    return arr.reshape(nrows, n, ncols, m).swapaxes(1, 2)


def parMatMult(a, b, nblocks, mblocks, dot_func=matMult):
    """
    Return the matrix product a * b.
    The product is split into nblocks * mblocks partitions that are performed
    in parallel threads.
    """
    n_jobs = nblocks * mblocks
    print('running {} jobs in parallel'.format(n_jobs))

    out = np.empty((a.shape[0], b.shape[1]), dtype=a.dtype)

    out_blocks = blockshaped(out, nblocks, mblocks)
    a_blocks = blockshaped(a, nblocks, 1)
    b_blocks = blockshaped(b, 1, mblocks)

#TODO We might need to fix this algorithm for correct results
    threads = []
    for i in range(nblocks):
        for j in range(mblocks):
            th = threading.Thread(target=dot_func,
                                  args=(a_blocks[i, 0, :, :],
                                        b_blocks[0, j, :, :],
                                        out_blocks[i, j, :, :]))
            th.start()
            threads.append(th)

    for th in threads:
        th.join()

    return out

def main():
    #if ( len ( sys.argv ) != 2 ) :
    #    sys.stderr.write ( "\nUsage:  mini-mpi_sweep_triclinic_hdf5.py config.txt\n" )
    #    sys.exit ( 0 )

    #read_config()  #TODO read configuration file in here
                   #Config file will have important configurations such as
                   #comm, size rank,
                   #name input, output folders and names


    #TODO Maybe get a config file to read name path comm size rank etc

    args = parse_args()
    print(args)

    root_path = args.data_root_dir + '/phase{}'.format(args.phase) + '/'
    print("root_path for data = ", root_path)

    comm = 1
    size = 1
    rank = 1
    print("Size is ", size)
    print("Rank is ", rank)
    print("Running pattern is ", args.exec_pattern)

    # Create project
    name = "test"
    path_in = "./"
    path_out = "./"
    name_out = "test.txt"

    symmetry = "cubic"  #Check if we need to emulate each symmetry for now it is will called cubic

    # Get ranges for sweeping parameters
    #rangeDict = ranges  #read ranges from config file

    # All symmetries sweep over a
    # Get start, stop and step
    #tuple_a = rangeDict['cell_1']
    # Generate array (evenly spaced values within the given interval)
    #a_range = np.arange(tuple_a[0], tuple_a[1], tuple_a[2])

    #For now To test generate 2 random matrix and multiply them
    np.random.seed(27)
    msz = args.mat_size
    A = np.random.randint(1,3000,size = (msz,msz))
    B = np.random.randint(1,3000,size = (msz,msz))
#    print(f"Matrix A:\n {A}\n")
#    print(f"Matrix B:\n {B}\n")

    #if rank == 0:
    #    print('tuple_a: ', tuple_a)

    # Configure sweep according to symmetry
    #if symmetry == 'cubic':
    #    # Configure 1D sweeping grid
    #    grid_ = (a_range,)
    #    # Set sweeping function for symmetry given
    #    sweepf_ = cubic_lattice

    #Single Thread Multiply the matrices
    if (args.exec_pattern == "single-thread"):
        print("Single Threaded execution")
        start = time.time()
        C = np.matmul(A,B)
        time_par = time.time() - start
        print('single thread: {:.2f} seconds taken'.format(time_par))
    elif (args.exec_pattern == "multi-thread"):
        print("Multi Threaded execution")
        start = time.time()
        C = parMatMult(A, B, args.inner_bsz, args.inner_bsz) 
        time_par = time.time() - start
        print('multi thread: {:.2f} seconds taken'.format(time_par))
    elif (args.exec_pattern == "MPI"):
        print('MPI is currently not supported!')
        sys.exit(1)
    else:
        print("exec_pattern has to be single-thread, multi-thread or MPI")
        sys.exit(1)

#    print(C)

    filename = root_path + 'all_X_data.npy'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print("Start to write to file ", filename)
    with open(filename, 'wb') as f:
        np.save(f, C)

    filename = root_path + 'all_Y_data.npy'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print("Start to write to file ", filename)
    with open(filename, 'wb') as f:
        np.save(f, C)

    #if rank == 0:
    #    print('----------------------------------------------------------')
    #    print('Number of simulations (%s): %d, size of histogram: %d' % (symmetry, nsim, histosz))

if __name__ == '__main__':
    main()

