# Changelog

## 0.2.0 (unreleased)

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
