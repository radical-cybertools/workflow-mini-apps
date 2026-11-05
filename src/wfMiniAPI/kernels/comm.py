"""Communication kernels (MPI / NCCL collectives)."""

from ._backends import *  # noqa: F403 -- cp, nccl, MPI, h5py exist only when installed
from .misc import get_device_module
from .registry import annotate_kernel


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
