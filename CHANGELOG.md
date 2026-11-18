# Changelog

## 0.2.0 (unreleased)

### Phase 3: real backends, packaging, release

- **Scope:** in-task telemetry is out of scope for wfMiniAPI. `MiniApp` records the run
  and verifies AsyncFlow's task-level telemetry; nothing inside a task is recorded.
  `annotate_kernel` only registers kernels. The design record is now revision 4.
- **Packaging:** `dragon` extra (RHAPSODY's Dragon backend, dragonhpc 0.15.0) and `plot`
  extra; `wfminiapi.RunStarted` records the git commit of editable installs.
- **`tools/plot_run.py`:** task-level plot of a run from its telemetry checkpoint.
- **Real Dragon validation:** `WFMINI_EQUIV_BACKEND=dragon python
  tests/equivalence/run_equiv.py pair` runs the baseline and this tree on the same node
  and compares them (multisets). Every workflow matches on one node, dry and real.
- **On Dragon, a failing example now exits.** Before, a task failure skipped the core's
  `flow.shutdown()` and the process never exited; `MiniApp`'s exit shuts the engine down.
- **Lifecycle tests on Dragon:** `WFMINI_TEST_DRAGON=1 pytest -m dragon tests/integration`.
- **`tools/validate_node.sh`:** validation on an HPC node with Dragon and a GPU
  (`docs/testing/hpc-validation.md`).
- **Docs:** after an `AttachError`, the engine and telemetry are still the script's to
  shut down.

### Phase 2: explicit edge and MiniApp

- **`wfMiniAPI.api.MiniApp(flow, telemetry)`**: the mini-app's contact point. It
  attaches to the user-built engine and AsyncFlow telemetry, records
  `wfminiapi.RunStarted`/`RunEnded`, always stops telemetry after the engine has shut
  down, and verifies the checkpoint (`app.report`, plus
  `<session>.wfminiapi.report.json`). It owns no exit codes or signal handlers.
- **`wfMiniAPI.api.require_dragon()`**: optional preflight with a launch hint.
- **Example edges** (motif_2, motif_3 ×2, motif_4, motif_6, dragon-mpi): unversioned
  `DragonExecutionBackend`, `flow.start_telemetry(...)`, and
  `async with MiniApp(flow, telemetry), flow.workflow_scope("<name>")`. The outer
  function is now `run_miniapp(cfg)` (was `workflow`) and calls
  `miniapp_workflow(cfg, flow, logger)`, which holds the unchanged core. Every workflow
  still matches its baseline exactly.
- **DeepDriveSim**: the same edge in `run_workflow.py`, as
  `tests/equivalence/ddsim/run_workflow.patch`.
- **Tests:** `contract_check.py` (each workflow attaches `MiniApp`); integration
  scenarios (failures, signals, attach errors, early stop); layering and
  framework-drift tests.

### Phase 1: baseline and equivalence harness

Phase 1 of the telemetry-first MiniApp redesign (`docs/design/miniapp-telemetry.md`):
baseline and equivalence harness. No workflow behaviour changes.

### Changed
- **Repository layout.** One package at the root (`pyproject.toml`, `src/wfMiniAPI/`).
  `cpp/`, `benchmarks/`, `tools/analyze/`, `docs/` and `tests/` moved to the top level;
  the Tutorial task scripts are in `examples/tutorial/`.
- **Kernels split by kind** into `wfMiniAPI.kernels` (`misc`, `io`, `comm`, `transfer`,
  `compute`, `registry`). `wfMiniAPI.kernel` re-exports every name it had before, in
  the same registration order, so existing imports keep working.
- **`import wfMiniAPI` has no side effects**; `kernel`, `kernels`, `registry` and `sim`
  load on first use.
- **Packaging:** `pyproject.toml`, Python ≥ 3.11, base dependency `numpy` only.
  Extras: `io`, `mpi`, `gpu`, `workflow` (AsyncFlow and RHAPSODY[telemetry], pinned by
  commit), `test`, `docs`, `dev`. `radical-pilot`, `dragonhpc` and `rhapsody-py[all]`
  are no longer installed by default.

### Fixed
- KR-01: importing the kernels no longer crashes without CuPy.
- KR-07 (import part): importing the kernels no longer prints to stdout.
- KR-06: `scatterAdd` works on NumPy ≥ 1.24 and uses in-range indices.

### Removed
- `examples/DeepDriveMD` (superseded by DeepDriveSim's `workflows/miniapps_workflow`).
- `setup.py`, `requirements*.txt`, `MANIFEST.in`, `VERSION`.

### Moved
- `examples/ExaLearn` → `examples/legacy/exalearn/` (old RADICAL-Pilot API; unmaintained).

### Added
- `tests/equivalence/`: static and dynamic equivalence checks with goldens for the six
  AsyncFlow examples, the Tutorial scripts and DeepDriveSim `0c4f0c2`.
- `tests/unit/`: kernel-surface tests. MkDocs site. CI workflow.
