"""Optional array, MPI and HDF5 backends, imported once for all kernel modules.

As before the split, a backend that is not installed leaves its name undefined
(``cp``, ``nccl``, ``MPI``, ``h5py``), and the matching ``*_AVAILABLE`` flag is False.
Importing this module still imports mpi4py when it is installed (KR-02, unchanged).
"""

__all__ = ["CUPY_AVAILABLE", "MPI4PY_AVAILABLE", "H5PY_AVAILABLE"]

try:
    import cupy as cp
    from cupy.cuda import nccl
    CUPY_AVAILABLE = True
    __all__ += ["cp", "nccl"]
except ImportError:
    CUPY_AVAILABLE = False

try:
    from mpi4py import MPI
    MPI4PY_AVAILABLE = True
    __all__ += ["MPI"]
except ImportError:
    MPI4PY_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
    __all__ += ["h5py"]
except ImportError:
    H5PY_AVAILABLE = False
