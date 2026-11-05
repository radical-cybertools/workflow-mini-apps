# SOE design: explicit, telemetry-first building blocks for `workflow-mini-apps` (revision 3)

!!! note
    Design record copied from the design review (revision 3). Links to the review's
    working papers and bug tickets are kept as plain names.


**Status:** design proposal. Nothing is implemented, and no repository file was changed.
**Revision:** 3, 2026-11-05. It replaces revision 2, which hid the backend and engine behind `wfm.engine()` / `wfm.run()`. Earlier revisions are archived in `soe_miniapp_workflow_api_agents/`.

**Code checked:**

| Code | Revision |
|---|---|
| `workflow-mini-apps` (package `wfMiniAPI`) | `55cd20f` |
| AsyncFlow **main** | `d97585e` |
| RHAPSODY **main** | `142f396` |
| DeepDriveSim | `0c4f0c2` |
| Dragon | 0.14.1 installed; RHAPSODY main pins 0.15.0. Claims checked on 0.14.1 only are marked **[D0.14]**. |

**How this was produced.**
- **Round 1:** two agents analysed the code, against the 0.6.0 releases (superseded).
- **Round 2:** both agents re-checked everything on the main branches, proposed designs and critiqued each other.
- **Round 3:** the architect agent checked the explicit-attach design of this revision in source, with a probe run against the main sources (normal, SIGINT and SIGTERM).

Papers and probes: `round2/`, `round3/`.

**Citations:**

| Key | File |
|---|---|
| `af/wm.py` | `radical.asyncflow/src/radical/asyncflow/workflow_manager.py` |
| `rh/…` | `rhapsody/src/rhapsody/…` |
| `rh/tm.py` | `rhapsody/src/rhapsody/telemetry/manager.py` |
| `kern.py` | `wfMiniAPI/src/wfMiniAPI/kernel.py` |

---

## 1. Owner decisions this design follows

| # | Decision |
|---|---|
| **H1** | **Hard rule: explicit building blocks.** Users see, construct and can swap every component in the external API: the execution backend (`DragonExecutionBackend`, `ConcurrentExecutionBackend`), `WorkflowEngine`, and AsyncFlow's telemetry (`flow.start_telemetry`). wfMiniAPI never hides them. Its telemetry is an explicit, visible extension of AsyncFlow's telemetry, which itself extends RHAPSODY's. |
| U1 | `examples/DeepDriveMD` is dropped; the DDSim mini-app replaces it. |
| U2 | Only `from rhapsody.backends import DragonExecutionBackend`; no `V3`/`Vx` names. |
| U3 | Design against the **main** branches of AsyncFlow and RHAPSODY. |
| U4 | **Workflow cores stay unchanged.** The new API plugs around existing AsyncFlow workflows; only telemetry integration and external-API changes the new work imposes are allowed. |
| U5 | Known workflow bugs are preserved bug-for-bug and documented. |
| U6 | Task bodies stay textually unchanged (same kernel calls, old names, old units). |
| U7 | Phase 2: in-task telemetry is integrated into RHAPSODY's existing telemetry, not built as a parallel system. |
| D-a | No `op(...)` lines in existing task bodies: existing workflows get kernel- and task-level telemetry; step operations are for new workflows. |
| D-b | The DeepDriveSim baseline is `0c4f0c2`; the EO1 instrumentation is replaced by the library's telemetry. |
| D-c | The backend is explicit; **no fallback**. |
| D-d | GPU kernels are timed with CUDA events on the stream, resolved once at task end. |
| E1 | wfMiniAPI telemetry attaches as an **`async with`** block that guarantees the stop order, also on exceptions and signals. |
| E2 | "Mandatory" means **contract plus verification**. CI fails a workflow that doesn't attach; each run verifies its own telemetry. Kernels never hard-fail without telemetry. |
| E3 | `asyncio.run` stays. wfMiniAPI owns no exit codes and no signal handlers; the extension **reports**, and the script decides what to do with the report. |

---

## 2. Summary

Mini-apps are built from explicit RADICAL components, and they stay that way. The integrator writes, and can swap, each one:

```python
backend   = await DragonExecutionBackend()                 # RHAPSODY execution backend
flow      = await WorkflowEngine.create(backend=backend)   # AsyncFlow engine
telemetry = await flow.start_telemetry(...)                # AsyncFlow telemetry (RHAPSODY's TelemetryManager)
async with MiniApp(flow, telemetry, ...) as app:          # wfMiniAPI: the mini-app's contact point
    await workflow_core(cfg, flow, logger)                 # the unchanged workflow
```

**`MiniApp` is the mini-app's single contact point.** It is a thin coordinator over components the user built and passed in, and it never constructs or hides them (H1). Telemetry is its first capability (`app.telemetry`); later capabilities attach to the same object instead of adding new top-level names. Today it adds, on top of AsyncFlow's own telemetry (task lifecycle events, spans, metrics):
- **Kernel telemetry from unchanged task bodies.** `annotate_kernel`, the decorator already on every kernel, returns a recording wrapper. The extension passes each task its identity (the AsyncFlow uid), so every `kern.*` call is recorded and attributed to its task.
- **One `TaskOutcome` per task.** It carries the exit code and error class, which AsyncFlow does not report.
- **The real process start and end of each task.** These are independent of Dragon's early RUNNING and late completion.
- **Records feed RHAPSODY's own stream.** An ingest adapter registered with RHAPSODY emits each task's records just before AsyncFlow reports the task finished. That ties every record to the task's RHAPSODY span, in the same checkpoint file and trace tree.
- **A guaranteed stop order.** On leaving the block, normally, on an exception or after a signal, it waits for engine shutdown, drains the task records, verifies, and then stops telemetry. This makes DeepDriveSim's MA-11 ordering bug impossible.
- **A report:** `telemetry_complete`, counts, and unobserved or late records.

**The workflow core is untouched and byte-identical.** Every motif changes at the edge only: the unversioned backend import, the wfMiniAPI import, two telemetry lines (`start_telemetry` and the `async with`), and one new line that splits `workflow` from `workflow_core` so the core is not re-indented (§6.1). DeepDriveSim changes only `run_workflow.py`. A three-part equivalence check proves nothing in the core changed (§10).

---

## 3. What changed

### 3.1 From revision 2

| Revision 2 | Revision 3 | Why |
|---|---|---|
| `flow = await wfm.engine()` hid the backend and engine | Backend, engine and `start_telemetry` stay explicit in the workflow | H1 |
| `wfm.run()` replaced `asyncio.run` and owned signals and exit codes (0/2/3/128+sig) | `asyncio.run` stays; AsyncFlow's signal handlers stay; the extension returns a **report** | E3 |
| wfMiniAPI minted `run_id` and owned a run directory | Identity is the user's `flow.uid`; directories are the user's `checkpoint_path` / `work_dir` | H1: the user owns the components |
| Telemetry mandatory by construction (the platform built the engine) | Mandatory by **contract plus verification** | E2: the user builds the engine |
| Callback hook installed before engine creation | **Public re-registration** after creation, forwarding to `flow.task_callbacks` | The engine already exists at attach; verified, 4 of 4 linked |
| "Exit 2 with a launch hint" when Dragon is missing | An optional, visible `require_dragon()` preflight (Q1) | E3 removes wfMiniAPI exit codes |
| Telemetry class (`MiniAppTelemetry`) as the user's contact point | **`MiniApp(flow, telemetry, ...)`** is the single contact point; telemetry is its first capability (`app.telemetry`, `app.report`) | The mini-app, not telemetry, is the domain object; later capabilities attach to it without new top-level names. It stays a thin coordinator over user-built components (H1). |
| Edge API module `wfMiniAPI.workflow` | Removed. Public modules are `wfMiniAPI.telemetry` (manager side) and `wfMiniAPI.task` (in-task, new workflows) | H1 |

**Unchanged from revision 2:**
- the main-branch facts (§4);
- the preserved bugs;
- kernel recording, including D-d;
- the RHAPSODY integration mechanism;
- the KR-fix policy;
- the equivalence proof;
- the upstream proposals (with three additions, §8.5).

### 3.2 From revision 1

The authoring API of revision 1 becomes the optional layer for **new** workflows only: `MiniApp`, `TaskContext`, `Params`, and typed snake_case kernels with byte units. Existing workflows plug in as they are (U4, U6). Revision 1 also assumed 0.6.0 workarounds, which main made unnecessary: a separate invocation id, and counters to catch events lost at stop.

---

## 4. Current state, re-checked on main

### 4.1 What main fixed

| Revision-1 constraint | On main | Effect |
|---|---|---|
| Task uid assigned after the call | **Fixed:** readable as `fut.uid` at call time (`af/wm.py:737-740`) | The AsyncFlow uid is the task identity |
| Uid counter reset at shutdown | **Fixed for one process** (`af/utils.py:4-11`) | Join key stays `(session_id, task_id)` |
| `stop()` loses the backlog after 5 s | **Fixed:** drains while the backlog shrinks (`rh/tm.py:321-368`) | Simpler stop |
| Dragon non-zero exit becomes `SystemExit` | **Fixed:** `exit_code` set, task FAILED (`rh/backends/execution/dragon.py:320-329`). Under AsyncFlow it surfaces as `RuntimeError(stderr)` (`af/wm.py:1884-1896`). | `TaskOutcome` records `exit_code` |
| Dragon callbacks off the event loop | **Fixed** (`dragon.py:286-288,331-340`) | Callback hook can emit and ingest in order |
| Pending components not cancelled at shutdown | **Fixed** (`af/wm.py:2102-2112`) | |

### 4.2 What still holds

| # | Fact | Evidence | Consequence |
|---|---|---|---|
| F1 | No in-task telemetry API, no context propagation | `rh/telemetry/__init__.py:26-58` | Phase 2 provides it (§8). |
| F2 | `emit()` is not thread-safe; Concurrent runs function tasks in client threads | `rh/tm.py:419-432`; `concurrent.py:42-44` | Tasks never call `emit()`; off-loop producers use `call_soon_threadsafe`. |
| UP-08 | `emit()` after `stop()` is silently dropped | `rh/tm.py:426-427` | Ingest everything before `stop()`; that is what the `async with` exit guarantees. |
| F6 | AsyncFlow's signal handlers call `shutdown()`, which never stops telemetry | `af/wm.py:339-383,2063-2135` | Handled by the `async with` exit (§6.4). |
| F7 | Plain or foreign futures passed to a task raise `TypeError` | `af/wm.py:1018-1094` | Nothing wraps futures or the engine. |
| F8 | Concurrent replaces the environment; Dragon passes an environment only through `process_template(s)`; native Dragon functions have none | `concurrent.py:182,209-225`; `dragon.py:444-446,456-458,471-493` | Executables get the context in their environment; function tasks through a wrapper. |
| UP-01/02/03/04/06/07/09 | Dragon early RUNNING, late completion; Concurrent cancel doesn't kill; resources ignored; `error_type="unknown"`; spans timed at dispatch; no `span_id` for late custom events | see the round-2 papers | Recorded or worked around, as before. |
| C1 | `from rhapsody.backends import DragonExecutionBackendV3` raises `ImportError` | `rh/backends/__init__.py:46-52` | Every motif changes this line to the unversioned name (U2). |

### 4.3 New findings on main

| # | Finding | Consequence |
|---|---|---|
| UP-12 | `shutdown()` cancels tasks with no cancel event, so their spans are never exported (`af/wm.py:1901-1915,2096-2112`) | Ticket. The extension emits `TaskOutcome(state="abandoned")`. |
| UP-13 | RHAPSODY's `service.name="rhapsody"` overrides `OTEL_SERVICE_NAME` (`rh/tm.py:235-239`) | Ticket. The user can pass the optional helper `MiniApp.otel_resource(name=...)` to `start_telemetry`. |
| UP-14 | Adapters are stopped after the drain, so they cannot send final records (`rh/tm.py:321-374`) | The extension drains its adapter before calling `stop()`. |
| UP-15 | AsyncFlow's `start_telemetry` lacks `drain_on_stop` and the `ResourceLayout` event | Ticket; defaults are acceptable. |
| UP-16 | Both mains declare version 0.6.0 | Pin by commit; capability probe. |
| N1 | `register_adapter` does not start an adapter added after `start()` (`rh/tm.py:499-506`) | The extension calls `adapter.start()`. |
| N3 | Adapters may emit events only, not spans (`rh/telemetry/adapters/base.py:21-23`) | In-task records are events; real child spans need an upstream change. |
| N10 | RHAPSODY keeps every event in memory for the whole session (`rh/tm.py:148,1136`) | Per-task kernel summaries by default. |
| N11 | RHAPSODY picks its resource adapter by `isinstance` and class name (`rh/tm.py:529-586`) | The backend is never proxied; hooks go on the instance. |
| N14 | DeepDriveSim's Dragon → Concurrent fallback never fires: the constructor raises `RuntimeError`, or `AssertionError` outside the launcher, not `ImportError` | Removing it preserves behaviour (D-c). |
| — | **[D0.14]** Dragon runs native function tasks in threads inside long-lived pool workers | Context is per thread; sink files are per task. Re-check on 0.15.0. |

---

## 5. Decisions made in this design

| Topic | Decision | Why |
|---|---|---|
| Core not re-indented by `async with` | **Split at the edge:** one new column-0 line, `async def workflow_core(cfg, flow, logger):`, above the first core line. Construction and the `async with` stay in `workflow(cfg)`, in their original order. | Keeps core lines byte-identical, keeps the prologue order (dragon-mpi's `MPI_Init` still runs before Dragon starts), and is exception-safe. The alternatives were rejected: moving construction into a new `main()` reorders logging and `MPI_Init`; a `start()`/`aclose()` pair is not exception-safe; indentation-insensitive comparison contradicts the exact check (D-a). Owner to confirm (Q2). |
| How the extension reaches the backend | `flow.backend` (public, the default backend); `backends=[...]` for engines with several | `_backends` is private (`af/wm.py:90,150-153`). |
| Callback hook | `backend.register_callback(wrapper)` after creation; the wrapper ingests, then forwards to `flow.task_callbacks` | One slot, read at call time by every backend; nothing caches the old callback (`rh/base.py:109-119`; probe 4/4 linked). |
| Submit hook | Instance override of `backend.submit_tasks`, which passes a shallow copy of each task dict with the context added | AsyncFlow looks it up on every submit (`af/wm.py:1688,1698`); adapter selection unaffected (probe). |
| Tasks submitted before attach | `AttachError` naming the uids (seen in public `flow.running` / `flow.resolved`); a task merely *called* before attach is still observed | Attaching late is a contract bug. Q3. |
| Task-span index (internal) | **Not a user-facing component.** `MiniApp`'s telemetry capability keeps it internally and attaches it only when `export_traceparent=True`, through RHAPSODY's private `_tracer_provider` (drift-tested), until the public `add_span_processor` (U-J) lands. Late records need no index: they keep `task_id`, and analysis joins them to the task span already in the checkpoint. | It is plumbing, not a building block (H1 covers backend, engine and telemetry). RHAPSODY's tracer provider is private and not global (`rh/tm.py:248-251`), and subscribers receive events without span ids (`rh/tm.py:1142-1167`), so there is no public way to build it after `start()`. |
| Checkpoint path | Passed to both `start_telemetry` and `MiniApp`; checked at attach | RHAPSODY keeps it private (`rh/tm.py:144`). Upstream request U-H. |
| Identity | `session_id = flow.uid`, the user's choice (default `asyncflow.session.<hex8>`) | H1 |
| Dragon unavailable | No fallback (D-c). An explicit, optional `wfMiniAPI.preflight.require_dragon()` raises `BackendUnavailableError` with a launch hint; the exit status stays the script's own. | E3 rules out a wfMiniAPI exit code. Q1. |

---

## 6. The explicit edge

### 6.1 motif_4, before → after

The same diff applies to motif_2, motif_3 (both) and motif_6; only the line numbers differ. Core lines 27–122 are byte-identical, including `await flow.shutdown()`, and `__main__` is unchanged.

```diff
--- examples/motif_4/miniapp_async.py   (55cd20f)
+++ examples/motif_4/miniapp_async.py   (integrated)
@@ -1,9 +1,10 @@
 import argparse, asyncio, yaml, random, logging
 import time

-from rhapsody.backends import DragonExecutionBackendV3
+from rhapsody.backends import DragonExecutionBackend

 from radical.asyncflow import WorkflowEngine
 from radical.asyncflow.logging import init_default_logger

 from wfMiniAPI import kernel as kern
+from wfMiniAPI.app import MiniApp
@@ -22,8 +23,13 @@ async def workflow(cfg):
     init_default_logger(logging.DEBUG)

     mp.set_start_method("dragon")
-    backend = await DragonExecutionBackendV3()
+    backend = await DragonExecutionBackend()
     flow = await WorkflowEngine.create(backend=backend)
+    telemetry = await flow.start_telemetry(checkpoint_path="telemetry")
+    async with MiniApp(flow, telemetry, checkpoint_path="telemetry"):
+        await workflow_core(cfg, flow, logger)
+
+async def workflow_core(cfg, flow, logger):

     @flow.function_task
     async def experiment():
```

**Swapping a component is a visible one-line change.** For example, `backend = await ConcurrentExecutionBackend()`.

**What the integrator gets:**
- AsyncFlow's and RHAPSODY's own events, spans and metrics;
- `RunStarted` and `RunEnded`;
- one `TaskOutcome` per task;
- `TaskProcess` (real start and end);
- `KernelSummary` per task and per kernel argument signature;
- a verified report, logged as one line through the `wfMiniAPI.telemetry` logger (WARNING if incomplete), so no `print` is added.

**A script that wants an exit code does it visibly:**

```python
    async with MiniApp(...) as app:
        await workflow_core(cfg, flow, logger)
    return app.report          # __main__: report = asyncio.run(workflow(cfg)); sys.exit(report.exit_hint)
```

### 6.2 dragon-mpi

The prologue keeps `import mpi4py; from mpi4py import MPI` (`:21-22`) before the backend, so `MPI_Init` runs in the same order as today. Core lines `:29-149` are byte-identical. The `sN = cfg[...]` lines, which the task-description defaults read at definition time, stay in the core, in the same scope as the task definitions. The manager-side kernel call (`:32`) now runs inside the block and is recorded at session scope.

```diff
-from rhapsody.backends import DragonExecutionBackendV3
+from rhapsody.backends import DragonExecutionBackend
 ...
+from wfMiniAPI.app import MiniApp
 ...
     mp.set_start_method("dragon")
-    backend = await DragonExecutionBackendV3()
+    backend = await DragonExecutionBackend()
     flow = await WorkflowEngine.create(backend=backend)
+    telemetry = await flow.start_telemetry(checkpoint_path="telemetry")
+    async with MiniApp(flow, telemetry, checkpoint_path="telemetry"):
+        await workflow_core(cfg, flow, logger)
+
+async def workflow_core(cfg, flow, logger):
```

### 6.3 DeepDriveSim `run_workflow.py`

`run_workflow.py` is entirely edge, so it may be re-indented. `miniapps_workflow.py`, `ddsim_manager.py`, the task scripts and `config.yaml` stay byte-identical.

```diff
 from radical.asyncflow import WorkflowEngine
+from wfMiniAPI.app import MiniApp
 ...
-    # --- Build backend and asyncflow ---           (Dragon attempt + an ImportError fallback that never fires)
+    # --- Build backend and asyncflow (explicit; no fallback, D-c) ---
+    if backend == "dragon":
+        from rhapsody.backends import DragonExecutionBackend
+        execution_backend = await DragonExecutionBackend()
+    elif backend == "concurrent":
+        from rhapsody.backends import ConcurrentExecutionBackend
+        execution_backend = await ConcurrentExecutionBackend()
+    else:
+        raise ValueError(f"DDSIM_BACKEND={backend!r}: expected 'dragon' or 'concurrent'")
+    asyncflow = await WorkflowEngine.create(execution_backend)

-    # --- Telemetry ---                              (optional start_telemetry)
+    # --- Telemetry (mandatory, E2) ---
+    if not cfg.get("telemetry", True):
+        logging.getLogger(__name__).warning("config 'telemetry: false' is ignored: telemetry is mandatory")
+    telemetry_dir = cfg.get("telemetry_dir", "telemetry-output")
+    telemetry = await asyncflow.start_telemetry(resource_poll_interval=0.05, checkpoint_path=telemetry_dir)
+    print("Started Asyncflow telemetry ...")
 ...
-    replicas = [...]; try: gather ... finally: close, summary, telemetry.stop(), asyncflow.shutdown(), pending sweep
+    async with MiniApp(asyncflow, telemetry, checkpoint_path=telemetry_dir):
+        replicas = [...]                              # unchanged statements, re-indented (edge file)
+        try: gather ... finally: close, summary, asyncflow.shutdown()   # telemetry.stop() removed: the block stops it
+
+    pending = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}   # MOVED after the block
+    if pending: ... cancel ...
```

**Notes for the DeepDriveSim owners:**
- **The pending-task sweep must move after the block.** `asyncio.all_tasks()` includes RHAPSODY's dispatch and checkpoint tasks (`rh/tm.py:280-287`). Cancelling them before the block exits makes `stop()` discard the queued events (`rh/tm.py:351-368`).
- **Removing the fallback changes nothing in practice:** it never fired (N14). An unknown `DDSIM_BACKEND` now fails explicitly.
- **`telemetry: false` is warned about and ignored** (Q4). The repository config already sets `telemetry: true`. `MiniAppsWorkflow` always receives the manager, and its `self.telemetry.emit(...)` feeds the same stream the extension verifies.
- **Behaviour that stays:** MA-08 (errors swallowed, "no errors" banner).
- **MA-11 is gone:** the block stops telemetry after `asyncflow.shutdown()`.

### 6.4 Lifecycle inside the block

1. **`__aenter__`:**
   - check that `telemetry.session_id == flow.uid` and that the checkpoint file exists;
   - raise `AttachError` if any task was already submitted;
   - install the two hooks (§8.2);
   - subscribe a counter;
   - register and start the ingest adapter;
   - emit `RunStarted` (versions, commits, backend class, D-d mode, kernel detail).
2. **Inside the block:** the unchanged core runs, including its own `flow.shutdown()`.
3. **`__aexit__`** runs on normal exit, on exception and after a signal. It never suppresses the exception:
   1. Wait (polling `backend.state()`, with a timeout) for the engine to reach SHUTDOWN. If it never does, call `flow.shutdown()`.
   2. Emit `TaskOutcome(state="abandoned")` for tasks with no terminal state (UP-12).
   3. Drain the adapter (UP-14).
   4. Emit `RunEnded`.
   5. Verify (§6.5).
   6. `telemetry.stop()`.
   7. Log the report line.

**Signals (E3)**, verified by probe for SIGINT and SIGTERM:
- AsyncFlow's own handler shuts the engine down and cancels the awaited futures, so `CancelledError` reaches `__aexit__`.
- The ordered stop completes, and the checkpoint gets its `SessionEnded`.
- The exit stays 1 with a traceback, as today, and `report.interrupted = True`.
- A second signal is ignored by AsyncFlow during the exit (`af/wm.py:374-376`).

### 6.5 Verification and the report

`report.telemetry_complete` is true only if all of these hold:
- the checkpoint file exists;
- for each `wfminiapi` event type, emitted = seen by the subscriber = lines in the checkpoint;
- task records written = ingested;
- exactly one `TaskOutcome` per task;
- a `TaskProcess` for every function task;
- no task submitted outside the hooks (`unobserved`);
- telemetry was not stopped before the block exited (`stopped_before_exit`).

The report also carries `interrupted`, counts, `late_records`, and the reasons for any incompleteness. `report.exit_hint` (0 if complete, 3 if incomplete) is a convenience the script may use; wfMiniAPI never exits by itself.

---

## 7. API (typed Python ≥ 3.11)

```python
# wfMiniAPI/app.py   (manager side; the mini-app's single contact point)
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

class MiniApp:
    """Thin coordinator over user-built components. Never constructs or hides flow, backend or telemetry."""
    def __init__(self, flow: "WorkflowEngine", telemetry: "TelemetryManager", /, *,
                 checkpoint_path: str | Path,
                 backends: Sequence["BaseBackend"] | None = None,   # default [flow.backend]
                 kernel_detail: Literal["summary", "sampled", "all"] = "summary",  # "all": equivalence suite only
                 gpu_timing: Literal["cuda_events", "launch"] = "cuda_events",     # D-d
                 on_sink_error: Literal["mark", "fail"] = "mark",
                 export_traceparent: bool = False,   # only this needs the internal task-span index
                 shutdown_wait_s: float = 60.0) -> None: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, et, ev, tb) -> Literal[False]: ...
    flow: "WorkflowEngine"                 # the objects the user passed in, unchanged
    telemetry: "MiniAppTelemetry"          # the telemetry capability (wraps the user's TelemetryManager)
    report: "MiniAppReport"                # valid after the block
    @staticmethod
    def otel_resource(*, name: str) -> "Resource": ...   # optional: pass to start_telemetry(resource=...) (UP-13)

# wfMiniAPI/telemetry.py   (internal module behind app.telemetry; imports RHAPSODY)
class MiniAppTelemetry:
    manager: "TelemetryManager"            # the user's RHAPSODY/AsyncFlow telemetry, as passed in
    # internal: ingest adapter, the two backend hooks, task-span index, verification

class MiniAppReport:                       # frozen dataclass
    session_id: str
    telemetry_complete: bool
    interrupted: bool
    tasks: "TaskCounts"                    # created, done, failed, cancelled, abandoned, unobserved
    records_written: int
    records_ingested: int
    late_records: int
    reasons: tuple[str, ...]
    exit_hint: int                         # 0 complete, 3 incomplete; never applied by wfMiniAPI

class AttachError(Exception): ...

# wfMiniAPI/preflight.py   (optional, explicit)
def require_dragon() -> None: ...          # raises BackendUnavailableError with a launch hint (Q1)

# wfMiniAPI/task.py   (task side, stdlib only; NEW workflows only, D-a)
def op(name: str, **attrs: str | int | float | bool): ...        # context manager
def metric(name: str, value: float, *, unit: str) -> None: ...
def current_task() -> "TaskTelemetryContext | None": ...
```

**The kernel library keeps its current surface.** `wfMiniAPI.kernel` and `wfMiniAPI.registry` keep their names and signatures; `annotate_kernel` now returns the recording wrapper (§8.3).

---

## 8. Phase 2: in-task telemetry integrated into RHAPSODY

### 8.1 Options evaluated

| Option | Verdict |
|---|---|
| **A. RHAPSODY ingest adapter** reading task-side records and emitting RHAPSODY events | **Chosen.** Uses only public extension points (`TelemetryAdapter`, `register_adapter`, `define_event`, `make_event`). Records land in the same queue, checkpoint and trace tree as RHAPSODY's own events. |
| B. `TRACEPARENT` so tasks export OTel spans themselves | Rejected as a transport: it needs the OTel SDK and network access in every task, and could change OTel-instrumented code. Kept as **opt-in metadata**. |
| C. Tasks call `TelemetryManager.emit` | Impossible: client-only and not thread-safe (F1, F2). |
| D. Dragon-native channel | Dragon only; a later sink if a machine lacks a shared filesystem (Q3). |
| E. A RHAPSODY in-task API | The right **end state**; proposed upstream (U-E) using the same wire format, so adopting it later changes nothing for workflows. |

### 8.2 How it works

```
manager process (one asyncio loop) — every component constructed by the user
 ├ backend      = DragonExecutionBackend() | ConcurrentExecutionBackend()
 ├ flow         = WorkflowEngine.create(backend)
 ├ telemetry    = flow.start_telemetry(...)                     ──► one checkpoint, one trace tree
 └ MiniApp(flow, telemetry, ...)  (async with) → app.telemetry
      ├ TaskRecordAdapter registered with telemetry (RHAPSODY adapter API)
      ├ hook 1: backend.register_callback(wrapper)   — public re-registration after create()
      │         terminal state → adapter.drain(task_uid) → TaskOutcome → flow.task_callbacks(...)
      └ hook 2: backend.submit_tasks (instance override) — shallow copy of each task dict, plus the context:
                function task → partial(_run_with_context, fn, ctx)   (ContextVar set in the worker)
                executable    → env WFMINI_TASK_CONTEXT=<json>        (Concurrent: merged with inherited env;
                                                                       Dragon: process_template.env)
                native Dragon function → never given a template; malformed descriptors → left as they are

task (thread / process / Dragon worker / executable)
   ContextVar or WFMINI_TASK_CONTEXT → kernel recorder (annotate_kernel wrapper) → sink
   sink: in-memory (Concurrent thread tasks) | JSONL <checkpoint_path>/tasks/<session_id>/<task_uid>/<host>.<pid>.<tid>.jsonl
```

**Why the ordering matters.** RHAPSODY attaches a custom event to a task's span only while the task is active (UP-09). Hook 1 ingests the task's records **before** handing the terminal state to AsyncFlow, so every record arrives while the span is active. Records that arrive later anyway are counted as `late`. They keep their `task_id`, so analysis and the offline OTLP converter join them to the task span already recorded in the checkpoint; no index is needed for that. The only runtime use of a task-span index is the opt-in `TRACEPARENT`, and `MiniApp`'s telemetry capability creates that index internally when asked (§5).

**Why hooks on the instance.** RHAPSODY picks its adapter by type (N11), so the backend cannot be proxied. Both hooks live in one module with framework drift tests. They are deleted when upstream U-D lands.

### 8.3 Kernel recording (no task-body edits)

- `annotate_kernel` returns a `functools.wraps` recorder and registers it. With no context, the cost is one ContextVar read (about 0.2 µs).
- Only the outermost kernel call is recorded. Arguments are recorded as called, with derived fields: `nbytes_effective` (4 × `data_size` for the copy kernels), the resolved I/O path, and `read_bytes`.
- **Detail levels:**
  - `summary` (default): one `KernelSummary` per task × kernel × argument signature;
  - `sampled`: the first 3 calls per signature, then every 10th, as `KernelCall`;
  - `all`: equivalence suite only, because RHAPSODY keeps all events in memory (N10).
- **GPU (D-d):**
  - CuPy CUDA event pairs bracket each GPU kernel call on the current stream, with no synchronisation.
  - Completed pairs are folded in without blocking (`Event.done`) at summary checkpoints.
  - Pairs still open are resolved with one `synchronize()` at task end.
  - Host launch time is recorded alongside.
  - At most 4,096 pending pairs, then recorded sampling.
  - **Unverified:** there is no GPU here; the nightly GPU job must confirm there is no per-kernel sync.
- Manager-side kernel calls are recorded at session scope.

### 8.4 Events (`wfminiapi/1`)

All events are defined with RHAPSODY's `define_event` and use the in-task time as `event_time`.

| Event | Content |
|---|---|
| `wfminiapi.RunStarted` | versions and commits, backend class, D-d mode, kernel detail, config values the preserved bugs depend on |
| `wfminiapi.TaskOutcome` | one per task: state (done/failed/cancelled/abandoned), `exit_code`, error class, requested resources (recorded, not enforced), `cancel_verified`, how the context was injected |
| `wfminiapi.TaskProcess` | real start and end, host, pid, thread, `mpi_init_at_import` |
| `wfminiapi.KernelSummary` / `KernelCall` | per §8.3 |
| `wfminiapi.TelemetryHealth` | records written / ingested / late, sink errors |
| `wfminiapi.Op`, `wfminiapi.Metric` | new workflows only (D-a) |
| `wfminiapi.RunEnded` | counts, `telemetry_complete`, `interrupted` |

The wire format (task → manager) is EO1's validated JSONL, with `"v":1` on every line.

### 8.5 Local now, upstream in parallel

| Id | Repository | Proposal | If declined |
|---|---|---|---|
| U-C | RHAPSODY | adapter `flush()` before the drain; `register_adapter` starts the adapter when telemetry is running; a thread-safe `emit_threadsafe` | explicit drain, start, `call_soon_threadsafe` |
| UP-15 | AsyncFlow | `drain_on_stop` passthrough; `ResourceLayout` | defaults |
| UP-13 | RHAPSODY | let `OTEL_SERVICE_NAME` win | optional `MiniApp.otel_resource()` |
| U-B | RHAPSODY | bounded cache of completed span contexts (UP-09) | emit-before-terminal; late records joined by `task_id` after the run |
| **U-J** (new) | RHAPSODY | public `TelemetryManager.add_span_processor()` usable after `start()` | private `_tracer_provider` access, drift-tested, only when `export_traceparent=True` |
| **U-G** (new) | AsyncFlow | public read-only `WorkflowEngine.backends` | `backends=[...]` passed by the user |
| **U-H** (new) | RHAPSODY | public `TelemetryManager.checkpoint_path` | path passed twice, checked at attach |
| **U-I** (new) | AsyncFlow | a telemetry **extension** hook: `start_telemetry(extensions=[...])`, whose `stop()` stops extensions first | the `async with` block. With U-I, the stop order is owned by AsyncFlow's telemetry itself, and the extension stays explicit in the same call. |
| U-D | RHAPSODY | backends propagate `RHAPSODY_TASK_CONTEXT` and offer an `on_terminal(uid)` hook | the two instance hooks |
| U-E | RHAPSODY | `rhapsody.telemetry.task`: in-task API plus built-in ingest adapter, same wire format | wfMiniAPI's own probe and adapter |
| U-A | RHAPSODY | span-shaped task operation events → real child spans with explicit times | events, plus an offline converter to OTLP child spans |
| U-F | RHAPSODY | bound the in-memory event log | `summary` default |
| UP-12 / UP-06 | AsyncFlow | `TaskCanceled` on shutdown; real `error_type` / `exit_code` | `TaskOutcome` |
| UP-16 | both | tag releases | commit pins |

**Order:** U-C, UP-15, UP-13, U-G, U-H and U-J first (small); then U-B and U-I; then U-D and U-E; then U-A.

**If everything is declined,** the local layer is the implementation. Only two things are lost: live child spans for in-task records (an offline converter adds them after the run), and `span_id` on late records.

---

## 9. Layering and packaging

| Layer | Module | May import |
|---|---|---|
| L0 | `wfMiniAPI._probe` (context, recorder, sinks, wire format), `wfMiniAPI.task` (public in-task API) | stdlib only |
| L1 | `wfMiniAPI.kernel`, `wfMiniAPI.registry` (kernel bodies unchanged) | L0, numpy |
| L2 | `wfMiniAPI.app` (`MiniApp`, the public contact point), `wfMiniAPI.telemetry` (`MiniAppTelemetry` behind `app.telemetry`; internal: adapter, hooks, task-span index), `wfMiniAPI.preflight` | public RHAPSODY and AsyncFlow names |
| L4 (later) | authoring API for new workflows (revision 1 §5–§7) | — |

A subprocess test asserts that importing L0 and L1 loads no AsyncFlow, RHAPSODY, OpenTelemetry, Dragon or mpi4py, and prints nothing.

**Packaging:**
- `pyproject.toml`, Python ≥ 3.11.
- Base: `numpy`, `radical.asyncflow` and `rhapsody-py[telemetry]`, pinned by commit (`d97585e`, `142f396`) until tagged (UP-16).
- Extras: `[gpu]`, `[mpi]`, `[io]`, `[dragon]` (→ `rhapsody-py[dragon]`, `dragonhpc==0.15.0`).
- Removed: `radical-pilot`, base `dragonhpc`, `rhapsody-py[all]`.
- At `__aenter__`, a capability probe (public behaviour only) raises `AttachError` naming any missing capability: `fut.uid` at call time, and `drain_on_stop` in the `TelemetryManager` signature.

---

## 10. Proving the workflows are unchanged

| Check | What it compares | Pass condition |
|---|---|---|
| **E1, static** | Every core segment (task functions, task graph and steering code) is byte-identical, now inside `workflow_core`. Whole-file hashes for DeepDriveSim core files, Tutorial scripts and configs. The diff lies only inside declared edge spans (`edge.toml`), whose allowed additions are the import lines, the telemetry lines, the `async with` and the `workflow_core` def line. The file constructs `MiniApp(` with its `flow` and the returned `telemetry` (E2). | identical; contract present |
| **E2–E4, dynamic** | Baseline = the unedited commit plus a test-only `sitecustomize` (V3 alias, `set_start_method` no-op, independent kernel recorder). Both sides run in determinism mode (Concurrent, one worker, seeded `random`). Compared: the task graph; the ordered kernel trace from the **independent** recorder (task identity normalised by `__name__`, since `__qualname__` gains `workflow_core`); files written; the failure point (for the hard-coded GPU calls on CPU, the same error at the same call). | identical; the integrated run produces a report with no unobserved tasks |
| **(b) completeness** | The library's own `all`-mode records vs the independent recorder | identical |
| **(c) library neutrality** | The baseline again on the **baseline library** (`55cd20f` + KR-01) | identical |
| **Task-dict delta** | What the hooks changed in each task description | only `function` (wrapped) and `env` / `process_template.env` carrying `WFMINI_TASK_CONTEXT` (and `TRACEPARENT` if opted in) |
| **DeepDriveSim** | Command multisets per task kind, per-script kernel traces, structural invariants, plus an exact run on a reduced, seeded config | identical |

**Schedule:**
- every change: Concurrent backend, CPU;
- nightly: Dragon 0.15.0 and a GPU node, which also verifies D-d, pickling of the wrapped native function, and environment layering.

### 10.1 Kernel-library fixes under the "unchanged" rule

| Fix | Status |
|---|---|
| KR-01 CuPy guard; KR-06 `scatterAdd` dtype (unused in scope); KR-07a import-time prints; KR-05 `read_bytes` observation; the recording wrapper | **Allowed** (Phase 0/2) |
| KR-02 (rank from the launcher environment, mpi4py only in `*WithMPI`), lazy CuPy, lazy h5py | **Gated by calibration:** ship only if the task wall-time shift is within noise (Q5) |
| KR-03, KR-04 (GPU sync), strict KR-05, KR-07b (prints inside I/O kernels) | **Not in this work.** Opt-in "measured-behaviour" release |

### 10.2 Preserved bugs (U5)

Each is recorded, not fixed:
- 4 × `dataCopyH2D` size;
- H2D where D2H was meant;
- hard-coded `device="gpu"`;
- shared `./data_0.h5`;
- dragon-mpi not running MPI;
- DeepDriveSim MA-01/03/04/05/08;
- the 6 MB read of a 3.5 MB file.

Where and how each is recorded: revision 2 §6.1, archived.

---

## 11. Roadmap

| Phase | Contents | Exit criterion |
|---|---|---|
| **0. Baseline and harness** | E1 manifest and `edge.toml`; equivalence harness and goldens; CI; commit pins; `pyproject.toml`; allowed KR fixes; calibration for the gated ones. **Delete `examples/DeepDriveMD` (U1).** | E1 green; goldens committed; `import wfMiniAPI.kernel` works on a CPU host |
| **1. Explicit edge + manager-side telemetry** | `MiniApp` with its telemetry capability (attach checks, `RunStarted`/`RunEnded`, `TaskOutcome`, abandoned outcomes, ordered exit, verification, report), `preflight.require_dragon`. Edge changes in the six examples and DeepDriveSim `run_workflow.py`. | E1–E4 identical; SIGINT/SIGTERM reach the ordered exit with `RunEnded` and `SessionEnded` present; telemetry always stops after engine shutdown |
| **2. In-task telemetry integrated into RHAPSODY** | L0 probe (context, recorder, sinks, wire v1, D-d); recording `annotate_kernel`; KR-05 observation; adapter and the two hooks; `TaskProcess`, `KernelSummary`, `KernelCall`; emit-before-terminal; offline OTLP converter. Upstream proposals opened in parallel. | propagation 1.000; 100% span-linked for records ingested before the terminal event; check (b) passes; overhead within noise; Dragon and GPU nightly green |
| **3. Upstream adoption** | Use U-I (AsyncFlow extension hook) when it lands, so the block becomes optional; switch to `rhapsody.telemetry.task` and delete the hooks when U-D/U-E land; real child spans with U-A | edges stay explicit; hooks gone |
| **4. Opt-in** | Measured-behaviour release; the L4 authoring API with `wfMiniAPI.task.op` / `metric` for new workflows | explicit re-baseline |

**EO1 parity (D-a, D-b).** For existing workflows, parity is checked at task and kernel level: real task start and end, per-kernel totals per task, propagation, and overhead confidence intervals. EO1's step-level operations are retired for these workflows.

---

## 12. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | The two instance hooks rely on semi-public backend behaviour | One module, drift tests, commit pins; removed by U-D |
| R2 | A user attaches late, stops telemetry early, or reuses a backend in a second engine (which overwrites the callback) | `AttachError`; `stopped_before_exit` and `unobserved` in the report; CI contract check |
| R3 | No shared filesystem on a multi-node machine | Attach-time visibility check; Dragon sink later (Q3) |
| R4 | Memory growth in RHAPSODY's event log (N10) | `summary` default; U-F |
| R5 | CUDA event pairs accumulate on long GPU tasks (DeepDriveSim can make ~500k GPU calls per task) | Non-blocking folding; 4,096-pair cap with recorded sampling |
| R6 | Dragon 0.15.0 differs from 0.14.1 | Nightly Dragon job before Dragon edges are declared equivalent |
| R7 | DeepDriveSim's pending-task sweep, if left before the block exit, discards telemetry | The diff moves it; the equivalence check asserts `SessionEnded` and full propagation |

---

## 13. Open questions

| # | Question | Recommendation |
|---|---|---|
| Q1 | D-c asked for "exit 2 with a launch hint" when Dragon is unavailable, but E3 rules out wfMiniAPI exit codes. Is an explicit, optional `wfMiniAPI.preflight.require_dragon()` (raises with the hint; exit status stays the script's own) enough? | Yes. |
| Q2 | Accept the `workflow` / `workflow_core` split (one new def line per motif; the tasks' `__qualname__` changes, which nothing uses)? | Yes: it is the only option that keeps the core byte-identical, the prologue order intact and the block exception-safe. |
| Q3 | Is a shared filesystem guaranteed for the checkpoint directory on Delta, Polaris and Frontier? | Confirm per machine; otherwise bring the Dragon sink into Phase 2. |
| Q4 | DeepDriveSim `telemetry: false`: warn and ignore, or raise? | Warn and ignore. |
| Q5 | Ship KR-02 (environment-rank variant), lazy CuPy and lazy h5py in Phase 0 if calibration shows no shift beyond noise? | Yes if within noise; otherwise Phase 4. |
| Q6 | Push U-I (AsyncFlow telemetry extensions) early? It lets AsyncFlow's telemetry own the stop order while the extension stays explicit in `start_telemetry(extensions=[...])`. | Yes, alongside U-C. |
| Q7 | Do any in-scope tasks use non-default CUDA streams? | Assume the current stream; record it; the nightly GPU job verifies. |

---

## Appendix: evidence

- Round 3: `round3/1_architect_attach_verification.md` and probe `round3/probes/v4.py`. Covers attach after creation, the callback and submit hooks, tasks called before attach, signals, and the re-indent analysis with the full diffs.
- Round 2: `round2/` (four papers and probes `v1.py`–`v3.py`; the superseded revision 2 is `0_merged_doc_round2_superseded.md`).
- Round 1: `round1/` (0.6.0-based; its authoring-API design is the basis for the later L4 layer).
- Tickets: `bugs_wfminiapi.md`, `bugs_deepdrivesim.md`, `bugs_upstream_rhapsody_asyncflow_dragon.md`. UP-12 to UP-16 are **not yet added** to the upstream file.
