# SOE design: explicit, telemetry-first building blocks for `workflow-mini-apps` (revision 4)

!!! note
    Design record copied from the design review (revision 4). Links to the review's
    working papers and bug tickets are kept as plain names.


**Status:** Phases 1 and 2 are implemented on `feature/miniapp-telemetry` (`d29206e`, `2bfdca2`); Phase 3 (real backends, packaging, release) is in progress.

**Revision 4, 2026-11-06.** In-task telemetry is dropped (decision I1). This revision describes the design as built. Revision 3, which still planned a kernel recorder, task-context hooks and a RHAPSODY ingest adapter, is archived as `rev3_superseded.md`; older revisions are in `soe_miniapp_workflow_api_agents/`.

**Code checked:**

| Code | Revision |
|---|---|
| `workflow-mini-apps` (package `wfMiniAPI`) | baseline `55cd20f`; integrated on `feature/miniapp-telemetry` |
| AsyncFlow **main** | `d97585e` |
| RHAPSODY **main** | `142f396` |
| DeepDriveSim | baseline `0c4f0c2`; integrated through `tests/equivalence/ddsim/run_workflow.patch` |
| Dragon | 0.15.0 |

**Findings per phase:** `phase1_findings.md`, `phase2_findings.md`.

**Citations:**

| Key | File |
|---|---|
| `af/wm.py` | `radical.asyncflow/src/radical/asyncflow/workflow_manager.py` |
| `rh/…` | `rhapsody/src/rhapsody/…` |
| `rh/tm.py` | `rhapsody/src/rhapsody/telemetry/manager.py` |

---

## 1. Owner decisions this design follows

| # | Decision |
|---|---|
| **H1** | **Hard rule: explicit building blocks.** Users see, construct and can swap every component in the external API: the execution backend (`DragonExecutionBackend`, `ConcurrentExecutionBackend`), `WorkflowEngine`, and AsyncFlow's telemetry (`flow.start_telemetry`). wfMiniAPI never hides them. |
| **I1** | **(2026-11-06) In-task telemetry is the user's.** wfMiniAPI records the run, its scopes and each task's lifecycle and outcome. It never instruments task bodies, passes no task context into tasks and adds no hooks on the backend. Task-context propagation, if wanted, belongs upstream in RHAPSODY (U-D). Supersedes U7 and D-d. |
| U1 | `examples/DeepDriveMD` is dropped; the DDSim mini-app replaces it. |
| U2 | Only `from rhapsody.backends import DragonExecutionBackend`; no `V3`/`Vx` names. |
| U3 | Design against the **main** branches of AsyncFlow and RHAPSODY. |
| U4 | **Workflow cores stay unchanged.** The API plugs around existing AsyncFlow workflows; only the edge changes. |
| U5 | Known workflow bugs are preserved bug-for-bug and documented. |
| U6 | Task bodies stay textually unchanged. |
| D-b | The DeepDriveSim baseline is `0c4f0c2`; EO1's step-level OpLog is retired. |
| D-c | The backend is explicit; **no fallback**. |
| E1 | wfMiniAPI attaches as an **`async with`** block that guarantees the stop order, also on exceptions and signals. |
| E2 | "Mandatory" means **contract plus verification**: CI fails a workflow that doesn't attach; each run verifies its own telemetry. |
| E3 | `asyncio.run` stays. wfMiniAPI owns no exit codes and no signal handlers; it **reports**, and the script decides. |
| M1 | **`MiniApp(flow, telemetry)` is the single contact point.** Telemetry is one capability of it, not the whole API. |
| M2 | Running N workflows is the workflow developer's job, with AsyncFlow scopes and blocks; `MiniApp` does not orchestrate workflows. |

---

## 2. Summary

Mini-apps are built from explicit RADICAL components. The integrator writes, and can swap, each one:

```python
backend   = await DragonExecutionBackend()                        # RHAPSODY execution backend
flow      = await WorkflowEngine.create(backend=backend)          # AsyncFlow engine
telemetry = await flow.start_telemetry(checkpoint_path="telemetry")  # AsyncFlow telemetry (RHAPSODY's TelemetryManager)
async with MiniApp(flow, telemetry), flow.workflow_scope("motif_4"):
    await miniapp_workflow(cfg, flow, logger)                     # the unchanged workflow
```

**`MiniApp` is a thin coordinator** over components the user built and passed in (H1). On top of AsyncFlow's own telemetry (task lifecycle events, spans, metrics) it adds:
- **Attach checks.** The telemetry belongs to this flow, is running and writes a checkpoint, and no task was submitted yet. Otherwise `AttachError` says how to fix it.
- **Run events.** `wfminiapi.RunStarted` (versions and commits, backends) and `wfminiapi.RunEnded` (interrupted, exception type, duration).
- **A guaranteed stop order.** On leaving the block, normally, on an exception or after a signal: engine shut down, a FIFO barrier so every event is processed, `RunEnded`, then `telemetry.stop()`. DeepDriveSim's MA-11 ordering bug can no longer happen.
- **A verified report.** It is computed from the checkpoint file alone:
  - `telemetry_complete`;
  - task counts: completed, failed, canceled, abandoned, not submitted;
  - the reasons for any gap.

  It is written as `<session>.wfminiapi.report.json` and logged as one line.

**What it does not do (I1):** anything inside a task. There are no kernel timings and no in-task spans.

**The workflow core is untouched and byte-identical.** Every example changes at the edge only (§6). DeepDriveSim changes only `run_workflow.py`. The equivalence harness proves the cores are unchanged (§9).

---

## 3. What changed from revision 3

| Revision 3 | Revision 4 (as built) | Why |
|---|---|---|
| Kernel recorder in `annotate_kernel`, task context, sinks, RHAPSODY ingest adapter, `TaskProcess` / `KernelSummary`, CUDA-event timing (D-d) | **Dropped.** `annotate_kernel` only registers kernels, as before | I1 |
| Callback hook and `submit_tasks` hook on the backend instance | **None** | I1; outcomes come from AsyncFlow's own events (Phase 2 Q1) |
| `TaskOutcome` event per task | **None.** Counts come from `TaskCreated` / `TaskSubmitted` / terminal events | Phase 2 Q1 |
| Internal task-span index, `export_traceparent` | **None** | Nothing needs it without in-task records |
| `MiniApp(flow, telemetry, checkpoint_path=…, kernel_detail=…, gpu_timing=…, on_sink_error=…)` | `MiniApp(flow, telemetry, /, *, backends=None, shutdown_wait_s=60.0)`. The checkpoint path is read from the telemetry manager | Pass the path once; the in-task options are gone |
| `wfMiniAPI.app`, `wfMiniAPI.task` | `wfMiniAPI.api` (`MiniApp`, `MiniAppReport`, `AttachError`, `require_dragon`, `BackendUnavailableError`); no task-side module | `src/` layout of Phase 1; I1 |
| `async with MiniApp(...)` around `workflow_core` | `async with MiniApp(flow, telemetry), flow.workflow_scope("<name>"):` around `miniapp_workflow`; the outer function is `run_miniapp` | Scope ids tag every task span; owner's naming |
| Layer L0 (task-side probe) | Removed | I1 |

---

## 4. Current state on main

### 4.1 What main fixed (and this design relies on)

| Fix | Evidence |
|---|---|
| Task uid readable as `fut.uid` at call time | `af/wm.py:737-740` |
| Engine-scoped uid counter | `af/utils.py:4-11` |
| `stop()` drains while the backlog shrinks (`drain_on_stop`) | `rh/tm.py:321-368` |
| Dragon non-zero exit sets `exit_code` and FAILED | `rh/backends/execution/dragon.py:320-329` |
| Pending components cancelled at shutdown | `af/wm.py:2102-2112` |

### 4.2 What still holds

| # | Fact | Consequence |
|---|---|---|
| UP-08 | `emit()` after `stop()` is silently dropped (`rh/tm.py:426-427`) | `MiniApp` checks the running flag before its exit barrier (`stopped_before_exit`). |
| F6 | AsyncFlow's signal handlers call `shutdown()`, which never stops telemetry (`af/wm.py:339-383`) | Handled by the `async with` exit (§6.3). |
| F7 | Plain or foreign futures passed to a task raise `TypeError` | Nothing wraps futures or the engine. |
| UP-01 | Dragon and Concurrent emit `TaskStarted` at submission, not at the real start | Recorded; "running" in plots means "handed to the backend". Upstream ticket. |
| UP-06 | `TaskFailed` has no real `exit_code` / `error_type` | Upstream ticket. |
| C1 | `DragonExecutionBackendV3` no longer exists on main | Every example uses the unversioned name (U2). |

### 4.3 Findings on main that shape the implementation

| # | Finding | Consequence |
|---|---|---|
| UP-12 | `shutdown()` cancels running tasks with no terminal event | Reported as `abandoned`, not as telemetry loss. |
| UP-13 | `service.name="rhapsody"` overrides `OTEL_SERVICE_NAME` | Ticket. |
| UP-15 | AsyncFlow's `start_telemetry` lacks `drain_on_stop` passthrough | Defaults are acceptable; attach check requires RHAPSODY's `drain_on_stop`. |
| UP-16 | Both mains declare version 0.6.0 | Pinned by commit; `RunStarted` records commits. |
| UP-17 | Nested scopes and blocks don't nest their spans | DeepDriveSim tasks carry the replica block id, not `ddsim`. |
| UP-18 | OTel ids and exemplars draw from the global `random` | Neutralised in the test shim only; ticket. |
| N14 | DeepDriveSim's Dragon → Concurrent fallback never fires | Removing it preserves behaviour (D-c). |
| P1 | `flow.block` around an unchanged core breaks it (the core's own `flow.shutdown()` cancels the block) | Use `flow.workflow_scope`. |

---

## 5. Decisions made in this design

| Topic | Decision | Why |
|---|---|---|
| Core not re-indented | **Split at the edge:** one new column-0 line, `async def miniapp_workflow(cfg, flow, logger):`, above the first core line. Construction and the `async with` stay in `run_miniapp(cfg)`, in their original order. | Core lines stay byte-identical; prologue order is kept (dragon-mpi's `MPI_Init` still runs before Dragon starts); exception-safe. |
| How `MiniApp` reaches the backend | `flow.backend`; `backends=[...]` for engines with several | `_backends` is private. Upstream U-G. |
| Tasks submitted before attach | `AttachError` naming the uids (`flow.running` / `flow.resolved`) | Attaching late is a contract bug. |
| Checkpoint path | Read from the manager's private `_checkpoint_path` (drift-tested) | Pass it once; upstream U-H makes it public. |
| Telemetry running | Read from the private `_running` (drift-tested) | Detects an early `telemetry.stop()` in about 1 s instead of a 30 s barrier timeout. U-H. |
| Task completeness | From AsyncFlow's own events, tasks only (`block.` uids excluded) | No hook needed (Phase 2 Q1). |
| Exit ordering | A FIFO barrier event, awaited with a timeout, before `RunEnded` and `stop()` | The subscriber lags behind the emitter at exit. |
| Identity | `session_id = flow.uid` | H1 |
| Dragon unavailable | No fallback (D-c). Optional `require_dragon()` raises `BackendUnavailableError` with a launch hint; the exit status stays the script's own. | E3 |

---

## 6. The explicit edge

### 6.1 motif_4, before → after

The same change applies to motif_2, motif_3 (both), motif_6 and dragon-mpi; only the line numbers and the scope id differ.

```diff
-from rhapsody.backends import DragonExecutionBackendV3
+from rhapsody.backends import DragonExecutionBackend
 ...
 from wfMiniAPI import kernel as kern
+from wfMiniAPI.api import MiniApp
 ...
-async def workflow(cfg):
+async def run_miniapp(cfg):
     import multiprocessing as mp
     logger = logging.getLogger(__name__)
     init_default_logger(logging.DEBUG)
     mp.set_start_method("dragon")
-    backend = await DragonExecutionBackendV3()
+    backend = await DragonExecutionBackend()
     flow = await WorkflowEngine.create(backend=backend)
+    telemetry = await flow.start_telemetry(checkpoint_path="telemetry")
+    async with MiniApp(flow, telemetry), flow.workflow_scope("motif_4"):
+        await miniapp_workflow(cfg, flow, logger)
+
+async def miniapp_workflow(cfg, flow, logger):
     ...                                   # core, byte-identical, ends with await flow.shutdown()
 ...
-    asyncio.run(workflow(cfg))
+    asyncio.run(run_miniapp(cfg))
```

**Swapping a component is a visible one-line change**, for example `backend = await ConcurrentExecutionBackend()`.

**A script that wants an exit code does it visibly:**

```python
    async with MiniApp(flow, telemetry) as app, flow.workflow_scope("motif_4"):
        await miniapp_workflow(cfg, flow, logger)
    return app.report          # __main__: sys.exit(asyncio.run(run_miniapp(cfg)).exit_hint)
```

### 6.2 DeepDriveSim `run_workflow.py`

`run_workflow.py` is entirely edge. `miniapps_workflow.py`, `ddsim_manager.py`, the task scripts and `config.yaml` stay byte-identical. The reference change is `tests/equivalence/ddsim/run_workflow.patch`:
- **Backend:** explicit, no fallback, with `require_dragon()` in the Dragon branch. An unknown `DDSIM_BACKEND` fails explicitly.
- **Telemetry is mandatory.** `telemetry: false` is warned about and ignored.
- **The block:** `async with MiniApp(asyncflow, telemetry), asyncflow.workflow_scope("ddsim"):` around the replicas. `telemetry.stop()` is removed, because the block stops it.
- **The pending-task sweep moves after the block.** `asyncio.all_tasks()` includes RHAPSODY's dispatch and checkpoint tasks; cancelling them before the block exits makes `stop()` discard queued events.
- **Behaviour that stays:** MA-08 (errors swallowed, "no errors" banner). MA-11 is gone.

### 6.3 Lifecycle inside the block

1. **`__aenter__`:**
   - attach checks (§2);
   - subscribe an O(1) counter;
   - attach barrier;
   - emit `RunStarted`.
2. **Inside the block:** the unchanged core runs, including its own `flow.shutdown()`.
3. **`__aexit__`** runs on normal exit, on exception and after a signal. It never raises and never replaces the workflow's exception.
   1. **Shutdown:**
      - if the backend is already SHUTDOWN, wait until `flow.components` is empty;
      - on `CancelledError` (signal path), wait for AsyncFlow's own shutdown, with a timeout;
      - otherwise call `flow.shutdown()`.
   2. **Exit barrier:** if telemetry is no longer running, record `stopped_before_exit`; otherwise emit the barrier and wait for it.
   3. Emit `RunEnded`.
   4. Call `telemetry.stop()`. This always runs, inside a `finally`.
   5. Verify the checkpoint, write the report file and log one line.

**Signals (E3):** AsyncFlow's own handler shuts the engine down and cancels the awaited futures, so `CancelledError` reaches `__aexit__`. The ordered stop completes, the checkpoint gets `RunEnded` and `SessionEnded`, and the exit stays 1 with a traceback, as before. On the Concurrent backend the exit waits for the running thread task to finish (Phase 2 Q5).

### 6.4 The report

`telemetry_complete` is true only if all of these hold:
- exactly one checkpoint file exists for the session;
- for each `wfminiapi` event type, emitted = seen by the subscriber = written;
- every AsyncFlow event type the subscriber saw is in the file;
- telemetry was not stopped before the block exited.

**Task counts** are reported, not judged:
- `created`, `completed`, `failed`, `canceled`;
- `abandoned` (submitted, no terminal event, UP-12);
- `not_submitted` (e.g. a failed dependency).

`report.exit_hint` (0 complete, 3 incomplete) is a convenience the script may use; wfMiniAPI never exits by itself.

---

## 7. API (Python ≥ 3.11)

```python
# wfMiniAPI/api: the mini-app's public surface
class MiniApp:
    """Thin coordinator over user-built components. Never constructs or hides flow, backend or telemetry."""
    def __init__(self, flow, telemetry, /, *, backends=None, shutdown_wait_s: float = 60.0): ...
    async def __aenter__(self) -> "MiniApp": ...
    async def __aexit__(self, et, ev, tb) -> bool: ...   # always False
    flow: "WorkflowEngine"
    telemetry: "MiniAppTelemetry"       # the telemetry capability (wraps the user's TelemetryManager)
    report: "MiniAppReport | None"      # valid after the block

class MiniAppReport:                    # frozen dataclass
    session_id: str
    checkpoint: str | None
    telemetry_complete: bool
    interrupted: bool
    tasks: "TaskCounts"                 # created, completed, failed, canceled, abandoned, not_submitted
    events: dict[str, int]              # wfminiapi events written, by type
    reasons: tuple[str, ...]
    exit_hint: int                      # property: 0 complete, 3 incomplete

class AttachError(RuntimeError): ...
class BackendUnavailableError(RuntimeError): ...
def require_dragon() -> None: ...       # optional, explicit preflight

# wfMiniAPI/telemetry: internal (capability, events, verify_checkpoint)
```

**The kernel library keeps its current surface.** `wfMiniAPI.kernel` keeps its names and signatures; `annotate_kernel` registers kernels and does nothing else.

---

## 8. Layering and packaging

| Layer | Module | May import |
|---|---|---|
| L1 | `wfMiniAPI.kernel`, `wfMiniAPI.kernels.*` (kernel bodies unchanged) | stdlib, numpy; CuPy, h5py, mpi4py where the kernel needs them |
| L2 | `wfMiniAPI.api`, `wfMiniAPI.telemetry` | L1, public AsyncFlow and RHAPSODY names (plus the two drift-tested private reads) |

A subprocess test asserts that importing L1 loads no AsyncFlow, RHAPSODY, OpenTelemetry or Dragon.

**Packaging:**
- `pyproject.toml` at the repository root, `src/` layout, Python ≥ 3.11.
- Base dependency: `numpy`.
- **Extras:**
  - `workflow`: AsyncFlow and `rhapsody-py[telemetry]`, pinned by commit until tagged (UP-16);
  - `dragon` (→ `rhapsody-py[dragon]`, `dragonhpc==0.15.0`);
  - `gpu`, `mpi`, `io`, `plot`, `test`, `docs`, `dev`.

---

## 9. Proving the workflows are unchanged

| Check | What it compares | Pass condition |
|---|---|---|
| **E1, static** | Core segments, located by anchors, are byte-identical to the baseline manifest. The diff lies only inside allowed edge lines (`tests/equivalence/edge.toml`). | identical |
| **E2, contract** | AST check: `run_miniapp` builds `flow` and `telemetry`, and `async with MiniApp(flow, telemetry), flow.workflow_scope(...)` encloses `await miniapp_workflow(...)` | present in every edge |
| **Dynamic, Concurrent** (CI) | The baseline and the integrated tree in determinism mode (one-worker Concurrent, reseeded `random`, OTel neutralised in the test shim). Compared: task calls and dependency edges, top-level kernel calls, files, exit code, failure point; DeepDriveSim as multisets. Telemetry facts asserted on the integrated run. | identical; complete report |
| **Dynamic, Dragon** (Phase 3, single node) | The same comparison under the `dragon` launcher, as multisets | identical; complete report |
| **GPU node** (Phase 3, `tools/validate_node.sh`, run by the owner) | Real mode on Dragon with a GPU, where the hard-coded GPU calls succeed | identical; complete report |

### 9.1 Kernel-library fixes under the "unchanged" rule

| Fix | Status |
|---|---|
| KR-01 CuPy guard; KR-06 `scatterAdd` dtype (unused in scope); KR-07a import-time prints | **Done** (Phase 1) |
| KR-02 (rank from the launcher environment), lazy CuPy, lazy h5py | **Gated by calibration:** ship only if the task wall-time shift is within noise (Q1) |
| KR-03, KR-04 (GPU sync), KR-05, KR-07b | **Not in this work.** Opt-in "measured-behaviour" release |

### 9.2 Preserved bugs (U5)

Recorded, not fixed:
- 4 × `dataCopyH2D` size;
- H2D where D2H was meant;
- hard-coded `device="gpu"`;
- shared `./data_0.h5`;
- dragon-mpi not running MPI;
- DeepDriveSim MA-01/03/04/05/08;
- the 6 MB read of a 3.5 MB file.

---

## 10. Upstream proposals

| Id | Repository | Proposal | Removes on our side |
|---|---|---|---|
| U-H | RHAPSODY | public `TelemetryManager.checkpoint_path` and `is_running` | both private reads and their drift tests |
| UP-06 | AsyncFlow | real `exit_code` and `error_type` on `TaskFailed` | — (better reports) |
| Block marker | AsyncFlow | a component type on `TaskCreated` | the `block.` uid-prefix rule in `verify_checkpoint` |
| UP-12 | AsyncFlow | a terminal event for tasks cancelled at shutdown | the `abandoned` category |
| UP-17 | AsyncFlow / RHAPSODY | nested scopes and blocks nest their spans | the DeepDriveSim scope exception |
| UP-18 | RHAPSODY | OTel ids and exemplars not drawn from the global `random` | the shim's OTel neutralisation |
| UP-01 | RHAPSODY | `TaskStarted` at the real start | — (correct "running" in plots) |
| U-G | AsyncFlow | public read-only `WorkflowEngine.backends` | the `backends=` argument for multi-backend engines |
| U-I | AsyncFlow | telemetry extensions in `start_telemetry(extensions=[...])`, stopped first by `stop()` | optional: the block could become a declared extension |
| UP-13, UP-15, UP-16 | both | service name, `drain_on_stop` passthrough, tagged releases | commit pins |

**Withdrawn with I1:** U-A (child spans for in-task operations), U-B (span cache for late records), U-C (adapter flush), U-D (task-context propagation), U-E (in-task API), U-F (bounded event log), U-J (span processor after `start()`).

---

## 11. Roadmap

| Phase | Contents | Status |
|---|---|---|
| **1. Baseline and harness** | Equivalence harness and goldens; `src/` layout, MkDocs, ExaLearn to `examples/legacy/`; safe KR fixes; `examples/DeepDriveMD` deleted | Done (`d29206e`) |
| **2. Explicit edge + `MiniApp`** | `MiniApp` with attach checks, run events, ordered exit, verification and report; `require_dragon()`; the edge in six examples and the DeepDriveSim patch | Done (`2bfdca2`) |
| **3. Real backends, packaging, release** | Dragon validation on one node; GPU-node validation script; `dragon` and `plot` extras; issue drafts; release preparation | In progress |
| **4. Upstream adoption** | Use U-H, UP-17, the block marker and U-I as they land; delete the matching workarounds | Later |
| **5. Opt-in** | Measured-behaviour release; an authoring API for new workflows (separate decision) | Later |

---

## 12. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | The two private reads (`_checkpoint_path`, `_running`) change upstream | Drift tests, commit pins; removed by U-H |
| R2 | A user attaches late, stops telemetry early, or passes another flow's telemetry | `AttachError`; `stopped_before_exit` in the report; CI contract check |
| R3 | Dragon 0.15.0 behaves differently from the Concurrent determinism mode | Phase 3 Dragon validation before release |
| R4 | DeepDriveSim's pending-task sweep, if left before the block exit, discards telemetry | The patch moves it; the equivalence check asserts a complete report |
| R5 | On signals, the exit waits for running tasks (Concurrent `executor.shutdown(wait=True)`) | Documented; framework behaviour (UP-03) |

---

## 13. Open questions

| # | Question | Recommendation |
|---|---|---|
| Q1 | Ship KR-02 (environment-rank variant), lazy CuPy and lazy h5py if calibration shows no shift beyond noise? | Yes if within noise; otherwise Phase 5. |
| Q2 | Push U-I (AsyncFlow telemetry extensions)? It would let AsyncFlow own the stop order while the extension stays explicit. | Yes, after U-H. |

---

## Appendix: evidence

- Revision 3 and its in-task design: `rev3_superseded.md`.
- Round 3: `round3/1_architect_attach_verification.md` and probe `round3/probes/v4.py`.
- Round 2: `round2/`; Round 1: `round1/`.
- Tickets: `bugs_wfminiapi.md`, `bugs_deepdrivesim.md`, `bugs_upstream_rhapsody_asyncflow_dragon.md`.
