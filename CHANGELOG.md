# Changelog

## 0.2.0 (unreleased)

### Runs of many engines and components

- **`app.part(name)`** hands another engine of the run a picklable `PartRef`; that
  engine's own `MiniApp(flow, telemetry, part_of=ref)`, in any process, reports into
  the run. The run's `verify()` requires a complete report from every part it handed out.
- **`app.component(name, files, verify=...)`** registers a component's files and a
  verifier returning a `ComponentReport`; **`app.check(name, verify=... | ok=...)`**
  registers a check across the run (a `Run`); **`app.parameters(...)`** records the
  run's configuration (`wfminiapi.Parameters` and the report).
- **The report** gains `run_id`, `part`, `parameters`, `completed_by_executable`,
  `parts`, `components`, `checks` and `checks_pass`. `exit_hint` is `4` when telemetry is
  complete but a check failed (`3` still means incomplete telemetry).
- **`wfminiapi.RunStarted`** gains `run_id` and `part`.
- **`python -m wfMiniAPI.verify <dir>`** verifies a run again offline from its manifest
  (`<session>.wfminiapi.manifest.json`, written only for runs that use the above).
- A run of one engine that uses none of this sees no change.

### Design B: the script owns the shutdown; MiniApp records and verifies

- **Breaking: new script shape.** The script that builds the engine and telemetry now
  shuts them down, in a nested `finally`, and calls the new `app.verify()`
  The nesting keeps telemetry stopped and the report written when the engine's
  shutdown fails or a second Ctrl-C arrives. A script still using only
  `async with MiniApp(...)` no longer gets telemetry stopped or a report, and on Dragon
  it may not exit: update it to this shape.
- **Why:** AsyncFlow #125 makes `flow.shutdown()` stop telemetry. An exit that shut the
  engine down and then recorded `RunEnded` would have lost it on every run. The script
  shape works with and without #125 (`telemetry.stop()` after it does nothing).
- **`MiniApp`'s exit** only records `ended_by` and queues `wfminiapi.RunEnded`; it
  awaits nothing and shuts nothing down. The F2 machinery (the shielded exit task in
  the caller's Context, absorbed cancels, `uncancel()`) is removed, as is the exit
  barrier. `RunEnded` now comes before the engine's shutdown events, and its
  `duration_s` covers the block, not the shutdown.
- **`app.verify()`** (new) reads the checkpoint, sets `app.report` and writes the report
  file. It never raises (F11 closed): telemetry still running, an `AttachError`, a
  block never entered, or an error while reading the checkpoint become reasons.
- **The report's new `notes`:** facts that do not make telemetry incomplete, e.g.
  `RunEnded` lost because the engine's shutdown stopped telemetry before `MiniApp`'s
  exit. Telemetry stopped by hand while the engine still ran stays a reason
  (`stopped_before_exit`), and so does a `telemetry.stop()` that did not finish writing
  its queue (no `SessionEnded` seen).
- **A WARNING at interpreter exit** if a script never calls `app.verify()`.
- **The workflow cores changed by one line:** each core's last line,
  `await flow.shutdown()`, moved to the script; DeepDriveSim's patch moves its shutdown
  out of the block the same way. The static check proves nothing else changed: its
  core boundary now ends before that line, and the manifest was rebuilt (commit A)
  from a tree that still passed against the baseline manifest, before the line moved
  (commit B). `static_check.py build` now reads moved files the way `check` does.
- **Contract check:** expects the new script shape.
- **New docs page `docs/telemetry.md`:** what a run records (events, metrics, spans,
  the report), the settings, how to read the data, and the limits.
- **Docs look:** the Material theme, laid out like the RHAPSODY docs, with a warm beige
  palette and each page's contents in the left sidebar (`mkdocs-material` added to the
  `docs` extra). The Kernels and Home pages are rewritten with sections.

### Hardening: the exit and attach fixes of the SOE review (F1, F2, F3, F9, F10)

- **`MiniApp`'s exit joins AsyncFlow's shutdown.** It is now one `await flow.shutdown()`,
  relying on AsyncFlow's joinable `shutdown()` (radical.asyncflow `fix/joinable-shutdown`,
  pinned by commit until merged): a shutdown the workflow, a signal or an engine error
  already started is waited for, not repeated. No more polling of backend states.
  - **F1:** a cancel from outside (`asyncio.timeout`, a `TaskGroup`, a supervisor) no
    longer waits 60 s for a signal shutdown that never started, and is no longer
    reported as a signal.
  - **F2:** a cancel that arrives during the exit no longer breaks it: the engine shuts
    down, then telemetry stops and the report is written, then the cancel is
    re-raised. If the workflow raised its own exception, that exception wins. On
    Dragon such a run used to hang.
  - **F10:** a backend kept with `flow.shutdown(skip_execution_backend=True)` stays
    running.
- **`shutdown_wait_s` is removed** (F9: it was never a bound). `MiniApp` sets no time
  limit; `docs/api/miniapp.md` explains how to bound a run.
- **The report's `ended_by` replaces `interrupted`:** `normal`, `exception`, `cancelled`
  or `engine_shutdown`, also in `wfminiapi.RunEnded`. The schema tag stays
  `wfminiapi/1`, which was never released; checkpoints written during development
  carry `interrupted` instead.
- **F3: new `AttachError` cases.** `telemetry` must be the manager the flow emits into
  (`flow.telemetry`): a manager built by hand, or one from an earlier
  `start_telemetry()` call, used to pass and report "complete" with no tasks. An
  AsyncFlow without the joinable shutdown is refused with an install hint.
- **Python ≥ 3.12** (was 3.11): the exit runs in the caller's Context
  (`Task.get_context()`), which RHAPSODY's OTel context detach needs.
- **Pins:** AsyncFlow `fix/joinable-shutdown` (on v0.7.0); RHAPSODY `142f396` (in
  v0.7.0). Both projects now tag releases.

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

Phase 1 of the telemetry-first MiniApp redesign (design record kept outside the repository):
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
