# Equivalence harness

The telemetry work must not change how any workflow behaves. `tests/equivalence/`
proves that, before and after every change.

## What is checked

| Check | What it compares |
|---|---|
| **Static** (`static_check.py`, `edge.toml`, `core_manifest.json`) | Every core segment of each example (helpers, prologue, tasks and task graph, `__main__`) is byte-identical; whole files for configs, Tutorial scripts and the DeepDriveSim core. Edge lines may change only as `edge.toml` allows. |
| **Dynamic** (`harness.py`, `run_equiv.py`, `goldens/`) | Each workflow runs unmodified under a test-only shim, and its summary must equal the golden: task/block calls and dependency edges, the ordered top-level kernel calls of each task with their arguments, manager-side kernel calls, files written, exit code and failure point. |

## Modes

- **dry**: kernels are recorded but not executed. The full task graph runs, with no
  I/O and no GPU failures.
- **real**: kernels execute, using the tiny configs in `configs/`. On a CPU host,
  tasks that hard-code `device="gpu"` fail at the same kernel call every time; when a
  run fails, only the executions up to the first failure are compared, because how
  many already-queued tasks AsyncFlow still starts after a failure is timing-dependent.
- **DeepDriveSim** runs in real mode only: its steering waits for files that dry mode
  would not write. It is compared as multisets (task kinds, commands, per-script kernel
  traces), because its steering order is timing-dependent.

## The shim (`shim/sitecustomize.py`)

Active only when the harness sets `WFMINI_EQUIV=1`. It records an independent trace
(it does not use the library's own telemetry), and makes the unmodified examples run
deterministically on a CPU host: `DragonExecutionBackendV3` becomes a one-worker
`ConcurrentExecutionBackend`, `set_start_method("dragon")` is skipped, and `random` and
NumPy are seeded. It is applied identically on both sides of every comparison.

## Usage

```shell
pip install -e ".[workflow,io,mpi,test]"
python tests/equivalence/static_check.py check
python tests/equivalence/run_equiv.py check            # all workflows, all modes
python tests/equivalence/run_equiv.py check --workflow motif_4 --mode dry --keep
pytest tests/equivalence
```

Goldens are captured on the **baseline**: workflow-mini-apps `55cd20f` plus the KR-01
CuPy guard, exported to `.equiv-work/wfmini-55cd20f`, and DeepDriveSim `0c4f0c2`
exported to `.equiv-work/ddsim-0c4f0c2`:

```shell
git archive 55cd20f | tar -x -C .equiv-work/wfmini-55cd20f
sed -i '321s/^)$/) if CUPY_AVAILABLE else None/' .equiv-work/wfmini-55cd20f/wfMiniAPI/src/wfMiniAPI/kernel.py
git -C ../DeepDriveSim archive 0c4f0c2 | tar -x -C .equiv-work/ddsim-0c4f0c2
python tests/equivalence/run_equiv.py capture
```

Tests that need an export are skipped when it is missing.
