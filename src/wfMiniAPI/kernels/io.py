"""I/O kernels (HDF5 files, with and without MPI)."""

import os

import numpy as np

from ._backends import *  # noqa: F403 -- cp, nccl, MPI, h5py exist only when installed
from .registry import annotate_kernel


@annotate_kernel
def writeSingleRank(num_bytes, data_root_dir):
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
    elif not H5PY_AVAILABLE:
        raise ImportError("h5py is not installed. Install h5py to use read/write.")
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            filename = os.path.join(data_root_dir, "data.h5")
            
            num_elem = num_bytes // 4
            data = np.empty(num_elem, dtype=np.float32)
    
            with h5py.File(filename, 'w') as f:
                dset = f.create_dataset("data", data = data)

@annotate_kernel
def writeNonMPI(num_bytes, data_root_dir, filename_suffix=None):
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
    elif not H5PY_AVAILABLE:
        raise ImportError("h5py is not installed. Install h5py to use read/write.")
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if filename_suffix == None:
            filename = os.path.join(data_root_dir, "data_{}.h5".format(rank))
        else:
            filename = os.path.join(data_root_dir, "data_{}_{}.h5".format(rank, filename_suffix))
        print("In writeNonMPI, rank = ", rank, " filename = ", filename)
        
        num_elem = num_bytes // 4
        data = np.empty(num_elem, dtype=np.float32)

        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset("data", data = data)

@annotate_kernel
def writeWithMPI(num_bytes, data_root_dir, filename_suffix=None):
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
    elif not H5PY_AVAILABLE:
        raise ImportError("h5py is not installed. Install h5py to use read/write.")
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        num_elem = num_bytes // 4
        num_elem_tot = num_elem * size
        data = np.empty(num_elem, dtype=np.float32)

        if filename_suffix == None:
            filename = os.path.join(data_root_dir, 'data.h5')
        else:
            filename = os.path.join(data_root_dir, "data_{}.h5".format(filename_suffix))
        print("In writeWithMPI, rank = ", rank, " filename = ", filename)

        with h5py.File(filename, 'w', driver='mpio', comm=comm) as f:
            dset = f.create_dataset("data", (num_elem_tot, ), dtype=np.float32)
            offset = rank * num_elem
            dset[offset:offset+num_elem] = data

@annotate_kernel
def readNonMPI(num_bytes, data_root_dir, filename_suffix=None):
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
    elif not H5PY_AVAILABLE:
        raise ImportError("h5py is not installed. Install h5py to use read/write.")
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if filename_suffix == None:
            filename = os.path.join(data_root_dir, "data_{}.h5".format(rank))
        else:
            filename = os.path.join(data_root_dir, "data_{}_{}.h5".format(rank, filename_suffix))
        print("In readNonMPI, rank = ", rank, " filename = ", filename)
        
        num_elem = num_bytes // 4

        with h5py.File(filename, 'r') as f:
            data = f['data'][0:num_elem] 

@annotate_kernel
def readWithMPI(num_bytes, data_root_dir, filename_suffix=None):
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
    elif not H5PY_AVAILABLE:
        raise ImportError("h5py is not installed. Install h5py to use read/write.")
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        num_elem = num_bytes // 4
        num_elem_tot = num_elem * size
        data = np.empty(num_elem, dtype=np.float32)

        if filename_suffix == None:
            filename = os.path.join(data_root_dir, 'data.h5')
        else:
            filename = os.path.join(data_root_dir, "data_{}.h5".format(filename_suffix))
        print("In readWithMPI, rank = ", rank, " filename = ", filename)

        with h5py.File(filename, 'r', driver='mpio', comm=comm) as f:
            dset = f['data']
            offset = rank * num_elem
            dset.read_direct(data, np.s_[offset:offset+num_elem])
