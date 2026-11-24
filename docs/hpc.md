# Run on HPC

This page shows how to run a mini-app on an HPC allocation. The script picks the
execution backend; the rest of the [MiniApp script](api/miniapp.md#the-script) does not
change. The backends and their settings are documented by RHAPSODY: see
[Execution](https://radical-cybertools.github.io/rhapsody/execution/).

The [examples](examples.md) use the Dragon backend, so the steps below show Dragon.

## Install

On the node, with the site's Python, CUDA and MPI modules loaded:

```shell
pip install -e ".[workflow,io,mpi,dragon,gpu]"
```

Leave out `gpu` on a node without an NVIDIA GPU. `gpu` installs `cupy-cuda12x`; for
another CUDA version, install the matching CuPy package instead.

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

**GPU kernels.** Several example tasks call kernels with `device="gpu"`. Without CuPy and
a GPU, those tasks fail with
`ImportError: CuPy is not installed. Install CuPy to use GPU capabilities.`, and the
report counts them as `failed`.

## Launch

Run the script with the `dragon` launcher, from the example's directory:

```shell
cd examples/motif_4
dragon -s miniapp_async.py      # one node
dragon miniapp_async.py         # every node of the allocation
```

Dragon reads the node count from SLURM. Outside a SLURM job, set it by hand:
`export SLURM_JOB_NUM_NODES=1`.

## Logging

The `dragon` launcher drops library log messages, including `MiniApp`'s summary line
and its warnings, unless the script configures logging. The examples do, with AsyncFlow's
`init_default_logger()`; in your own script, `logging.basicConfig(level=logging.INFO)`
is enough. The report file is written either way, and the `verify() was never called`
warning is always printed.

## After a killed run

A run killed from outside (walltime, `kill -9`) cannot shut Dragon down. Clean up before
the next run on the same allocation. On one node:

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
