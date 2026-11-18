# Getting started

## Install

Python 3.11 or newer.

```shell
git clone https://github.com/radical-cybertools/workflow-mini-apps.git
cd workflow-mini-apps
pip install -e ".[workflow,io,mpi]"
```

| Extra | Installs |
|---|---|
| `io` | `h5py` (I/O kernels) |
| `mpi` | `mpi4py` (MPI kernels) |
| `gpu` | `cupy-cuda12x` (GPU kernels) |
| `workflow` | AsyncFlow and RHAPSODY (with telemetry), needed by the examples |
| `dragon` | the Dragon backend (`dragonhpc` 0.15.0, through RHAPSODY); run scripts under the `dragon` launcher |
| `plot` | `matplotlib`, for `tools/plot_run.py` |
| `test`, `docs`, `dev` | pytest; MkDocs; ruff and pre-commit |

## Use the kernels

```python
import wfMiniAPI.kernel as wf

wf.matMulSimple2D("cpu", 1024)
wf.writeNonMPI(num_bytes=1 << 20, data_root_dir="./")
print(wf.list_kernels())
```

## Attach a workflow to MiniApp

Build the RADICAL components explicitly, then enter `MiniApp` around the workflow:

```python
from rhapsody.backends import ConcurrentExecutionBackend
from radical.asyncflow import WorkflowEngine
from wfMiniAPI.api import MiniApp

backend   = await ConcurrentExecutionBackend()
flow      = await WorkflowEngine.create(backend=backend)
telemetry = await flow.start_telemetry(checkpoint_path="telemetry")
async with MiniApp(flow, telemetry) as app, flow.workflow_scope("my_workflow"):
    ...                       # tasks, task graph, steering
    await flow.shutdown()
print(app.report.telemetry_complete)
```

`MiniApp` records the run and verifies that AsyncFlow's task telemetry reached the
checkpoint. It records nothing inside a task: in-task telemetry is yours. See
[MiniApp](api/miniapp.md).

## Plot a run

`tools/plot_run.py` draws one run from its telemetry checkpoint: the task timeline, the
tasks handed to the backend over time, node CPU and memory, and run time per task kind.

```shell
pip install -e ".[plot]"
python tools/plot_run.py examples/motif_4/telemetry run.png
```

The directory must hold one run. "Running" starts at AsyncFlow's `TaskStarted`, which
is emitted when a task is handed to the backend, not when it really starts.

## Run an example

Each example has a README with its configuration and launch command, for example
`examples/motif_4/README.md`.
