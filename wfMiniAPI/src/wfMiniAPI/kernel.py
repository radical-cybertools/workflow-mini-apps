import numpy as np
import time
import os
import sys

from .registry import annotate_kernel, list_kernels, kernel_params, run_kernel, time_kernel


print("Python executable location:", sys.executable)
print("NumPy version:", np.__version__)
print("NumPy location:", np.__file__)

try:
    import cupy as cp
    from cupy.cuda import nccl
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from mpi4py import MPI
    MPI4PY_AVAILABLE = True
except ImportError:
    MPI4PY_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


#################
#misc 
#################

@annotate_kernel
def sleep(seconds):
    time.sleep(seconds)

@annotate_kernel
def get_device_module(device):
    if device == "gpu":
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not installed. Install CuPy to use GPU capabilities.")
        return cp
    else:
        return np


#################
#io
#################

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


#################
#comm 
#################

@annotate_kernel
def MPIallReduce(device, data_size):
    xp = get_device_module(device)
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to perform allreduce.")
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        sendbuf = xp.empty(data_size, dtype=xp.float32)
        recvbuf = xp.empty(data_size, dtype=xp.float32)
        
        if device == "cpu":
            comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
    
        elif device == "gpu":
            uid = nccl.get_unique_id()
            comm_nccl = nccl.NcclCommunicator(size, uid, rank)
            comm_nccl.allReduce(sendbuf.data.ptr, recvbuf.data.ptr, data_size, nccl.NCCL_FLOAT32, nccl.NCCL_SUM, cp.cuda.Stream.null)
            cp.cuda.Stream.null.synchronize()
    
@annotate_kernel
def MPIallGather(device, data_size):
    xp = get_device_module(device)
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to perform allreduce.")
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        sendbuf = xp.empty(data_size, dtype=xp.float32)
        recvbuf = xp.empty(data_size * size, dtype=xp.float32)

        if device == "cpu":
            comm.Allgather(sendbuf, recvbuf)

        elif device == "gpu":
            uid = nccl.get_unique_id()
            comm_nccl = nccl.NcclCommunicator(size, uid, rank)
            comm_nccl.allGather(sendbuf.data.ptr, recvbuf.data.ptr, data_size, nccl.NCCL_FLOAT32, cp.cuda.Stream.null)
            cp.cuda.Stream.null.synchronize()


#################
#data movement
#################

@annotate_kernel
def dataCopyH2D(data_size):
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not installed. Install CuPy to use GPU capabilities.")
    else:
        data_h = np.empty(data_size, dtype=np.float32)
        data_d = cp.asarray(data_h)

@annotate_kernel
def dataCopyD2H(data_size):
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not installed. Install CuPy to use GPU capabilities.")
    else:
        data_d = cp.empty(data_size, dtype=cp.float32)
        data_h = cp.asnumpy(data_d)


#################
#computation
#################

@annotate_kernel
def matMulSimple2D(device, size):
    xp = get_device_module(device)
    matrix_a = xp.empty((size, size), dtype=xp.float32)
    matrix_b = xp.empty((size, size), dtype=xp.float32)
    matrix_c = xp.matmul(matrix_a, matrix_b)

@annotate_kernel
def matMulGeneral(device, size_a, size_b, axis):
    xp = get_device_module(device)
    matrix_a = xp.empty(tuple(size_a), dtype=xp.float32)
    matrix_b = xp.empty(tuple(size_b), dtype=xp.float32)
    matrix_c = xp.tensordot(matrix_a, matrix_b, axis)

@annotate_kernel
def fft(device, data_size, type_in, transform_dim):
    xp = get_device_module(device)
    if type_in == "float":
        data_in = xp.empty(tuple(data_size), dtype=xp.float32)
    elif type_in == "double":
        data_in = xp.empty(tuple(data_size), dtype=xp.float64)
    elif type_in == "complexF":
        data_in = xp.empty(tuple(data_size), dtype=xp.complex64)
    elif type_in == "complexD":
        data_in = xp.empty(tuple(data_size), dtype=xp.complex128)
    else:
        raise TypeError("In fft call, type_in must be one of the following: [float, double, complexF, complexD]")

    out = xp.fft.fft(data_in, axis=transform_dim)

@annotate_kernel
def fftn(device, data_size, type_in, transform_dim):
    xp = get_device_module(device)
    if type_in == "float":
        data_in = xp.empty(tuple(data_size), dtype=xp.float32)
    elif type_in == "double":
        data_in = xp.empty(tuple(data_size), dtype=xp.float64)
    elif type_in == "complexF":
        data_in = xp.empty(tuple(data_size), dtype=xp.complex64)
    elif type_in == "complexD":
        data_in = xp.empty(tuple(data_size), dtype=xp.complex128)
    else:
        raise TypeError("In fftn call, type_in must be one of the following: [float, double, complexF, complexD]")

    out = xp.fft.fftn(data_in, axes=transform_dim)

@annotate_kernel
def axpy(device, size):
    xp = get_device_module(device)
    x = xp.empty(size, dtype=xp.float32)
    y = xp.empty(size, dtype=xp.float32)
    y += 1.01 * x

#_axpy_inplace = cp.ElementwiseKernel(
#    'float32 alpha, raw float32 x',
#    'raw float32 y',               
#    'y[i] += alpha * x[i]',        
#    'axpy_inplace_kernel',
#    no_return=True                 
#)
#
#@annotate_kernel
#def axpy(device, size):
#    xp = get_device_module(device)
#    x = xp.empty(size, dtype=xp.float32)
#    y = xp.empty(size, dtype=xp.float32)
#    if xp == np:
#        y += 1.01 * x
#    elif xp == cp:
#        _axpy_inplace(1.01, x, y, size=size)

@annotate_kernel
def implaceCompute(device, size, num_op, op):
    xp = get_device_module(device)
    x = xp.empty(size, dtype=xp.float32)
    if isinstance(op, str):
        try:
            func = getattr(xp, op)
        except AttributeError:
            raise ValueError(f"The operator '{op}' is not available in the module {xp.__name__}.")
    elif callable(op):
        func = op
    else:
        raise ValueError("Operator must be either a string or a callable function.")
    for _ in range(num_op):
        x = func(x)

@annotate_kernel
def generateRandomNumber(device, size):
    xp = get_device_module(device)
    x = xp.random.rand(size)

@annotate_kernel
def scatterAdd(device, x_size, y_size):
    xp = get_device_module(device)
    y = xp.empty(y_size, dtype=xp.float32)
    x = xp.empty(x_size, dtype=xp.float32)
    idx = xp.empty(y_size, dtype=xp.int)
    if xp == np:
        y += x[idx]
    elif xp == cp:
        cp.add.at(y, idx, x)
