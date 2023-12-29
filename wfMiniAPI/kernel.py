import numpy as np
import time

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


def sleep(seconds):
    time.sleep(seconds)

def writeSingleRank(num_bytes):
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
    elif not H5PY_AVAILABLE:
        raise ImportError("h5py is not installed. Install h5py to use read/write.")
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            filename = "data.h5"
            
            num_elem = num_bytes // 4
            data = np.empty(num_elem, dtype=np.float32)
    
            with h5py.File(filename, 'w') as f:
                dset = f.create_dataset("data", data = data)


def writeNonMPI(num_bytes):
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
    elif not H5PY_AVAILABLE:
        raise ImportError("h5py is not installed. Install h5py to use read/write.")
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        filename = "data_{}.h5".format(rank)
        
        num_elem = num_bytes // 4
        data = np.empty(num_elem, dtype=np.float32)

        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset("data", data = data)

def writeWithMPI(num_bytes):
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

        with h5py.File('data.h5', 'w', driver='mpio', comm=comm) as f:
            dset = f.create_dataset("data", (num_elem_tot, ), dtype=np.float32)
            offset = rank * num_elem
            dset[offset:offset+num_elem] = data
#try to raise the error if file is ready
def readNonMPI(num_bytes):
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
    elif not H5PY_AVAILABLE:
        raise ImportError("h5py is not installed. Install h5py to use read/write.")
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        filename = "data_{}.h5".format(rank)
        
        num_elem = num_bytes // 4

        with h5py.File(filename, 'r') as f:
            data = f['data'][0:num_elem] 

def readWithMPI(num_bytes):
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

        with h5py.File('data.h5', 'r', driver='mpio', comm=comm) as f:
            dset = f['data']
            offset = rank * num_elem
            dset.read_direct(data, np.s_[offset:offset+num_elem])


def get_device_module(device):
    if device == "gpu":
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not installed. Install CuPy to use GPU capabilities.")
        return cp
    else:
        return np

def allReduce(device, data_size):
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
    
def allGather(device, data_size):
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


def dataCopyH2D(data_size):
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not installed. Install CuPy to use GPU capabilities.")
    else:
        data_h = np.empty(data_size, dtype=np.float32)
        data_d = cp.asarray(data_h)

def dataCopyD2H(data_size):
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not installed. Install CuPy to use GPU capabilities.")
    else:
        data_d = cp.empty(data_size, dtype=cp.float32)
        data_h = cp.asnumpy(data_d)


def matMulSimple2D(device, size):
    xp = get_device_module(device)
    matrix_a = xp.empty((size, size), dtype=xp.float32)
    matrix_b = xp.empty((size, size), dtype=xp.float32)
    matrix_c = xp.matmul(matrix_a, matrix_b)

def matMulGeneral(device, size_a, size_b, axis):
    xp = get_device_module(device)
    matrix_a = xp.empty(tuple(size_a), dtype=xp.float32)
    matrix_b = xp.empty(tuple(size_b), dtype=xp.float32)
    matrix_c = xp.tensordot(matrix_a, matrix_b, axis)

def axpy(device, size):
    xp = get_device_module(device)
    x = xp.empty(size, dtype=xp.float32)
    y = xp.empty(size, dtype=xp.float32)
    y += 1.01 * x

def implaceCompute(device, size, num_op):
    xp = get_device_module(device)
    x = xp.empty(size, dtype=xp.float32)
    for i in range(num_op):
        x = xp.sin(x)

def generateRandomNumber(device, size):
    xp = get_device_module(device)
    x = xp.random.rand(size)

#for the tutorial, three things:
#exalearn (CPU + GPU v1), ddmd v1, how to build wk-miniapp
#show installation script + run script, in installation script, show how to install assuming we are working in a brand new env (container for example)

