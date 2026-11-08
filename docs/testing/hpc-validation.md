# Validation on an HPC node

The equivalence suite in CI runs on the Concurrent backend on a CPU. Two things need a
real node:

- **Dragon:** task bodies run in Dragon worker processes, in parallel;
- **a GPU:** the examples' hard-coded GPU kernel calls succeed instead of failing.

`tools/validate_node.sh` checks both. For every workflow it compares the baseline
(workflow-mini-apps `55cd20f`, DeepDriveSim `0c4f0c2`) with this tree, both run now on
the same node under `dragon`. It also checks the MiniApp report of every integrated run,
then runs the MiniApp lifecycle scenarios on Dragon.

## Run it

On a compute node, from the repository root:

```shell
salloc ...                                   # one node, with a GPU
tools/validate_node.sh --ddsim ../DeepDriveSim
```

| Option | Meaning |
|---|---|
| `--ddsim PATH` | a DeepDriveSim clone; adds the DeepDriveSim workflow |
| `--python EXE` | interpreter for the venv (Python 3.11 to 3.13) |
| `--quick` | dry mode only, no lifecycle scenarios: checks that the script itself works |
| `--setup` | only build the venv (CuPy included) and export the baselines |
| `--offline` | install nothing; use the venv `--setup` made |
| `CUDA_PACKAGE=...` | the CuPy wheel for the node's CUDA (default `cupy-cuda12x`; `none` for no CuPy) |

### Compute nodes without internet

The venv needs internet (GitHub and PyPI). If the compute nodes have none, build it on a
login node first; the repository and its `.equiv-work/` must be on a filesystem both see.

```shell
# login node: load the same modules (Python, CUDA) you will use on the compute node
tools/validate_node.sh --setup --ddsim ../DeepDriveSim
# compute node, same modules
tools/validate_node.sh --offline --ddsim ../DeepDriveSim
```

The venv points at the interpreter that created it, so that interpreter must exist at the
same path on the compute node.

### MPI

The kernels import `mpi4py`, which needs an MPI library when it is imported. The script
checks this first and stops if there is none. Load the site's MPI module (the same one
for `--setup` and for the run), or install an MPI from PyPI into the venv where there is
internet: `.equiv-work/venv-validate/bin/pip install mpich`.

The script creates `.equiv-work/venv-validate`, and ends with

```
== result: PASS; send /…/validation-<host>-<date>.tar.gz
```

Send the tarball back. It holds the log, the versions (`pip freeze`, `nvidia-smi`), and
every run's summary, stdout, stderr and telemetry checkpoint.

## What passes

Each line of the comparison reads `identical, report complete`:

- **identical:** the baseline and the integrated run made the same task calls, ran the
  same kernels per task and wrote the same files (as multisets; Dragon runs tasks in
  parallel). A run that fails is compared at its failure point;
- **report complete:** the integrated run has one complete `MiniApp` report, one
  `RunStarted` and one `RunEnded`, and its tasks are tagged with the workflow scope.

Two notes may follow a result; neither is a failure:

- **"baseline hangs after the failure, integrated exits":** when a task fails, the
  baseline core never reaches its own `flow.shutdown()`. On Dragon nothing else shuts the
  backend down, so the baseline process never exits and is stopped after
  `WFMINI_EQUIV_TIMEOUT` (300 s). The integrated run exits, because `MiniApp` shuts the
  engine down. motif_2 always fails this way: it reads `./data_0.h5` before anything
  writes it (a preserved bug).
- **"attempt N, earlier ones hit the h5 lock race":** motif_2 and motif_4 tasks share
  `./data_0.h5` (a preserved bug), so on a parallel backend HDF5's file lock fails a task
  at random. Such runs are retried. If all 3 attempts race, the result is
  `INCONCLUSIVE`: rerun that workflow alone with
  `WFMINI_EQUIV_BACKEND=dragon .equiv-work/venv-validate/bin/python tests/equivalence/run_equiv.py pair --workflow <name>`.

## How the comparison stays deterministic on Dragon

The test shim (`tests/equivalence/shim/sitecustomize.py`) is loaded in every process,
including Dragon workers, and applied to the baseline and the integrated run alike:

- task bodies draw from their own `random` stream, seeded from the task's name and
  arguments. motif_2 and motif_6 steer on random values returned by tasks, and which
  worker runs a task is a matter of timing;
- the removed `DragonExecutionBackendV3` name is an alias of `DragonExecutionBackend`,
  so the baseline runs unmodified.
