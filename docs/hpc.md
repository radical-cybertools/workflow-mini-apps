# Run on HPC

This page shows how to run a mini-app on an HPC allocation. The script picks the
execution backend; the rest of the [MiniApp script](api/miniapp.md#the-script) does not
change. The backends and their settings are documented by RHAPSODY: see
[Execution](https://radical-cybertools.github.io/rhapsody/execution/).

The [examples](examples.md) use the Dragon backend, so the steps below show Dragon.

## Install

On the node, with the site's Python, MPI and GPU (CUDA or ROCm) modules loaded:

```shell
pip install -e ".[workflow,io,mpi,dragon,gpu]"
```

`gpu` installs `cupy-cuda12x`, the CuPy build for NVIDIA GPUs with CUDA 12. On a node
with another CUDA version, or with AMD GPUs (ROCm), leave out `gpu` and install the
matching CuPy build instead, for example `cupy-rocm-7-0` for ROCm 7.0. The kernels use
CuPy only when both of these imports work, so check once on the node:

```shell
python -c "import cupy; from cupy.cuda import nccl"
```

## In the script

The script builds the HPC backend where the [Quick start](getting-started.md#quick-start)
builds `ConcurrentExecutionBackend()`. With Dragon, the script also sets Dragon as the
multiprocessing start method first:

```python
import multiprocessing as mp
from rhapsody.backends import DragonExecutionBackend

mp.set_start_method("dragon")
backend = await DragonExecutionBackend()
```

**Check before you start (optional).** `require_dragon()` raises
`BackendUnavailableError` with a launch hint when Dragon is not installed or the script
was not started by the `dragon` launcher. `require_dragon()` never starts a Dragon
runtime, so it is cheap to call first:

```python
from wfMiniAPI.api import require_dragon

require_dragon()
```

**GPU kernels.** Every example has tasks that call GPU kernels in their code
(`device="gpu"`, `dataCopyH2D`). The `device` keys in `config.yaml` switch only some of
the other tasks, so setting them to `"cpu"` does not make an example run without a GPU.
Without CuPy and a GPU, the GPU tasks fail with
`ImportError: CuPy is not installed. Install CuPy to use GPU capabilities.`; the report
counts them as `failed`, and the tasks that depend on them as `not_submitted`.

## Launch

Run the script with the `dragon` launcher, from the example's directory:

```shell
cd examples/motif_4
dragon -s miniapp_async.py      # one node
dragon miniapp_async.py         # every node of the allocation
```

`dragon -s` runs on the current node and needs no workload manager. `dragon` without
`-s` runs on every node of the SLURM allocation it is started in; on a machine with
SLURM but outside an allocation, it stops with an error that suggests `-s`.

## Logging

The `dragon` launcher drops library log messages, including `MiniApp`'s summary line
and its warnings, unless the script configures logging. The examples do, with AsyncFlow's
`init_default_logger()`; in your own script, `logging.basicConfig(level=logging.INFO)`
is enough. The report file is written either way, and the `verify() was never called`
warning is always printed.

## After a killed run

A run killed from outside (walltime, `kill -9`) cannot shut Dragon down. Clean up before
the next run on the same allocation.

!!! warning "The cleanup stops all of your Dragon runs on those nodes"
    `dragon-cleanup` ends every Dragon runtime you own on the node, not only the killed
    run's. Do not run it while another of your Dragon runs is still active there.

On one node:

```shell
dragon-cleanup
```

On a multi-node allocation, run the cleanup once on every node:

```shell
srun --overlap -N$SLURM_NNODES --ntasks-per-node=1 dragon-cleanup -s
```

## Time limits

`MiniApp` sets none. For a hard limit, use the batch job's walltime or `timeout -k`; an
`asyncio.timeout()` around the `MiniApp` block bounds only the workflow, and the
shutdown in the script's `finally` still runs after it.
