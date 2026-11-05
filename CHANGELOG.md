# Changelog

## 0.2.0 (unreleased)

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
