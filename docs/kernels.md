# Kernels

This page lists the emulation kernels, the building blocks of a mini-app's tasks.

A kernel is a small, portable function that stands in for one kind of application
work: computation, file I/O, communication or host/GPU data movement. A mini-app task
calls a few of them, with sizes that match the real application, instead of running the
application itself; see [Write the workflow](building.md).

## Kernel families

### Compute

Module `wfMiniAPI.kernels.compute`. Runs on CPU (NumPy) or GPU (CuPy).

| Kernel | Stands in for |
|---|---|
| `matMulSimple2D(device, size)` | a square matrix product |
| `matMulGeneral(device, size_a, size_b, axis)` | a tensor contraction (`tensordot`) along chosen axes |
| `fft(device, data_size, type_in, transform_dim)`, `fftn(...)` | 1-D and n-D Fourier transforms |
| `axpy(device, size)`, `axpy_fast(device, size)` | vector updates `y = a*x + y` |
| `inplaceCompute(device, size, num_op, op)` | applying `op` (a NumPy/CuPy function name, or any callable) `num_op` times to one array |
| `generateRandomNumber(device, size)` | random number generation |
| `scatterAdd(device, x_size, y_size)` | indexed accumulation |
| `top_k(device, size, k)` | selecting the largest values |

### I/O

Module `wfMiniAPI.kernels.io`. Writes and reads HDF5 files of `num_bytes` per process.
Every I/O kernel needs `h5py` (`io` extra) and `mpi4py` (`mpi` extra): it reads its MPI
rank to name its file, also when it runs as a single process.

| Kernel | Stands in for |
|---|---|
| `writeSingleRank(num_bytes, data_root_dir)` | one writer: rank 0 writes `data.h5` |
| `writeNonMPI(num_bytes, data_root_dir, filename_suffix=None)` | each rank writing its own file, `data_<rank>.h5` |
| `writeWithMPI(num_bytes, data_root_dir, filename_suffix=None)` | processes writing one shared file through MPI |
| `readNonMPI(...)`, `readWithMPI(...)` | the matching reads |

### Communication

Module `wfMiniAPI.kernels.comm`. Needs `mpi4py` (`mpi` extra); on GPU it uses NCCL
collectives.

| Kernel | Stands in for |
|---|---|
| `MPIallReduce(device, data_size)` | an all-reduce across processes |
| `MPIallGather(device, data_size)` | an all-gather across processes |

### Data movement

Module `wfMiniAPI.kernels.transfer`. Needs CuPy (`gpu` extra).

| Kernel | Stands in for |
|---|---|
| `dataCopyH2D(data_size)` | copying data from host to GPU |
| `dataCopyD2H(data_size)` | copying data from GPU to host |

### Miscellaneous

Module `wfMiniAPI.kernels.misc`.

| Kernel | Does |
|---|---|
| `sleep(seconds)` | waits, for idle or fixed-length steps |
| `get_device_module(device)` | returns `numpy` or `cupy` (see [Devices](#devices)) |

The full reference, with each kernel's docstring, is the [Kernel API](api/kernels.md).

## Devices

Most kernels take a `device` argument:

- `"cpu"` runs them with NumPy;
- `"gpu"` runs them with CuPy, installed by the `gpu` extra (`cupy-cuda12x`, NVIDIA
  GPUs). Without CuPy a GPU call raises
  `ImportError: CuPy is not installed. Install CuPy to use GPU capabilities.`

`get_device_module(device)` returns the module a kernel uses (`numpy` or `cupy`), so a
task can allocate its own data on the same device.

## Using the kernels

```python
import wfMiniAPI.kernel as wf

wf.matMulSimple2D("cpu", 1024)                          # one 1024 x 1024 matrix product
wf.writeNonMPI(num_bytes=1 << 20, data_root_dir="./")   # write 1 MiB to ./data_0.h5 (needs h5py and mpi4py)
print(wf.list_kernels())                                # every kernel, in definition order
```

`wfMiniAPI.kernel` is the import path existing workflows use; it re-exports
`wfMiniAPI.kernels`. Importing it has no side effects: nothing is printed, and CuPy,
`mpi4py` and `h5py` are optional.

### Calling a kernel by name

The kernels are registered by name, so a configuration file can choose them:

| Function | What it does |
|---|---|
| `list_kernels()` | the names of all registered kernels |
| `kernel_params(name)` | the kernel's parameters, with their defaults and annotations |
| `run_kernel(name, **kwargs)` | call the kernel with keyword arguments |
| `time_kernel(name, device, n_warmup=3, n_repeat=20, **kwargs)` | run it `n_warmup` times, then time `n_repeat` runs; prints and returns the total, mean and standard deviation in milliseconds (GPU runs are timed with CUDA events) |

```python
total_ms, mean_ms, std_ms = wf.time_kernel("matMulSimple2D", "cpu", size=1024)
```

### Adding a kernel

Decorate a function with `annotate_kernel` to register it; it then appears in
`list_kernels()` and can be called with `run_kernel()`:

```python
from wfMiniAPI.kernels import annotate_kernel

@annotate_kernel
def my_stencil(device, size):
    ...
```

## The C++ kernels

`cpp/include/wfMiniAPI/kernel.h` is a header-only C++ version of the kernels, using
OpenMP target offloading, so it is meant for CPUs and NVIDIA, AMD and Intel GPUs. The
header is used by including it; there is nothing to install. Building with it needs OpenMP, MPI, HDF5
and NCCL headers at build time.

## Kernels and telemetry

A run's telemetry records tasks, not the kernels inside them. To time a kernel inside a
task, use `time_kernel()` or your own timers. See [Telemetry](telemetry.md).
