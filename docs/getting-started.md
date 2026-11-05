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
| `test`, `docs`, `dev` | pytest; MkDocs; ruff and pre-commit |

## Use the kernels

```python
import wfMiniAPI.kernel as wf

wf.matMulSimple2D("cpu", 1024)
wf.writeNonMPI(num_bytes=1 << 20, data_root_dir="./")
print(wf.list_kernels())
```

## Run an example

Each example has a README with its configuration and launch command, for example
`examples/motif_4/README.md`.
