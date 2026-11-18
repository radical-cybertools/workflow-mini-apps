# RADICAL Workflow Mini-Apps
Workflow Mini-Apps[^1] provides small, self-contained representations of scientific workflows (or mini-apps) for developing workflows.
Each mini-app is a simplified version of a complex scientific workflow, capturing its key tasks, data flow, and performance characteristics without the deployment challenges of the full application.
Workflow Mini-apps can be scaled and configured without application specific deployment challenges and constraints​.


Workflow Mini-app facilitate experimentation and helps understand workflow (distinct from application) performance.

## Repository layout

| Path | Contents |
|---|---|
| `src/wfMiniAPI/` | The Python package: emulation kernels (`wfMiniAPI.kernels`, also importable as `wfMiniAPI.kernel`) |
| `cpp/include/wfMiniAPI/` | Header-only C++ kernels |
| `examples/` | Workflow mini-apps on AsyncFlow and RHAPSODY: `motif_2`, `motif_3`, `motif_4`, `motif_6`, `dragon-mpi`; standalone task scripts in `tutorial/` |
| `examples/legacy/exalearn/` | The Neutron Diffraction (InverseProblem) mini-app on the old RADICAL-Pilot API; kept for reference, not maintained |
| `benchmarks/` | CUDA micro-benchmarks |
| `tools/` | `plot_run.py` (plot a run's telemetry), `validate_node.sh` (validation on an HPC node), `analyze/` (result-analysis helpers) |
| `docs/` | Documentation (MkDocs) |
| `tests/` | Unit tests and the workflow equivalence harness |

The AI-steered simulation mini-app (formerly `DeepDriveMD` here) now lives in
[DeepDriveSim](https://github.com/radical-collaboration/DeepDriveSim) (`workflows/miniapps_workflow`).

## Installation

Python 3.11 or newer.

```shell
git clone https://github.com/radical-cybertools/workflow-mini-apps.git
cd workflow-mini-apps
pip install -e ".[workflow,io,mpi]"    # add "gpu" for CuPy (CUDA 12), "test" for the tests
```

| Extra | Installs |
|---|---|
| `io` | `h5py` (I/O kernels) |
| `mpi` | `mpi4py` (MPI kernels) |
| `gpu` | `cupy-cuda12x` (GPU kernels) |
| `workflow` | AsyncFlow and RHAPSODY (with telemetry), needed by the examples |
| `dragon` | the Dragon backend (`dragonhpc` 0.15.0, through RHAPSODY) |
| `plot` | `matplotlib`, for `tools/plot_run.py` |
| `test`, `docs`, `dev` | pytest; MkDocs; ruff and pre-commit |

Each example's README explains how to run it. Every example attaches to
`wfMiniAPI.api.MiniApp`, which records the run and verifies AsyncFlow's task telemetry
(see `docs/api/miniapp.md`). Nothing inside a task is recorded: in-task telemetry is the
workflow developer's.

## Tests

```shell
pip install -e ".[workflow,io,mpi,test]"
pytest
```

`tests/equivalence/` checks that the example workflows behave exactly as their recorded
baseline; see its README. On a node with Dragon (and a GPU), `tools/validate_node.sh`
runs the same comparison on the real backend (`docs/testing/hpc-validation.md`).

This work has been supported by the DOE [RECUP](https://github.com/RECUP-DOE/workflow-miniapps) project.

[^1]: Ozgur O. Kilic, Tianle Wang, Matteo Turilli, Mikhail Titov, Andre Merzky, Line Pouchard, and Shantenu Jha (2024) "Workflow Mini-Apps: Portable, Scalable, Tunable & Faithful Representations of Scientific Workflows". https://doi.org/10.1109/CCGrid59990.2024.00059, https://arxiv.org/abs/2403.18073

