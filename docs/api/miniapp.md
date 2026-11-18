# MiniApp

`MiniApp` is a workflow mini-app's single contact point. You build the RADICAL
components yourself, and they stay visible and swappable: the execution backend,
the AsyncFlow engine and AsyncFlow telemetry. `MiniApp` attaches to them.

```python
from rhapsody.backends import DragonExecutionBackend
from radical.asyncflow import WorkflowEngine
from wfMiniAPI.api import MiniApp

backend   = await DragonExecutionBackend()
flow      = await WorkflowEngine.create(backend=backend)
telemetry = await flow.start_telemetry(checkpoint_path="telemetry")
async with MiniApp(flow, telemetry) as app, flow.workflow_scope("motif_4"):
    await miniapp_workflow(cfg, flow, logger)      # the unchanged workflow
print(app.report.telemetry_complete, app.report.tasks)
```

## What it does

**On entry**, it raises `AttachError` with a fix-it message if:

- `telemetry` is not the manager this `flow` emits into: pass the object the latest
  `flow.start_telemetry()` returned, not a manager built by hand, a wrapper, or the
  result of an earlier call;
- telemetry is not running, or has no checkpoint file (`checkpoint_path=`);
- tasks were already submitted (enter `MiniApp` right after `start_telemetry()`).

After an `AttachError`, `MiniApp` has not taken over anything: the engine and telemetry
are still running and remain yours to shut down (`await flow.shutdown()`,
`await telemetry.stop()`). On Dragon the process does not exit until both are.

Then it records `wfminiapi.RunStarted` (wfMiniAPI, AsyncFlow and RHAPSODY versions,
backend classes).

**On exit**, whether normal, on an exception, a cancel or a signal, it:

1. calls `await flow.shutdown()`. AsyncFlow shuts an engine down once: if the workflow,
   a signal or an engine error already started the shutdown, this waits for that same
   shutdown to finish. Backends you kept with `flow.shutdown(skip_execution_backend=True)`
   stay running; shut them down yourself (on Dragon the process does not exit until
   you do);
2. waits until every telemetry event so far has been processed;
3. records `wfminiapi.RunEnded`, then stops telemetry (after the engine; never before);
4. verifies the checkpoint file, and sets `app.report`.

The exit is not interrupted by a cancel. A cancel that arrives during it is delivered
once the steps above are done, so `asyncio.timeout`, `TaskGroup` and supervisors still
see it; if the workflow raised its own exception, that exception propagates instead.
Otherwise the exit does not raise: a failed engine shutdown becomes a reason in the
report. (An error inside the checkpoint verification itself can still escape; that is
finding F11, open.) Signals are handled by AsyncFlow, as before.

**Time limits.** `MiniApp` sets none: the exit lasts as long as the engine's shutdown
plus the telemetry drain (the event barrier waits up to 30 s). An `asyncio.timeout()`
around the run bounds the workflow, not the exit: its `TimeoutError` arrives after the
exit has finished, and Dragon's backend shutdown blocks the event loop while it runs.
For a hard limit use a process-level one (the batch walltime, `timeout -k`); after a
kill on Dragon, run `dragon-cleanup` and check for orphaned
`_dragon_native_python_process_main` processes.

## The report

`app.report` (`MiniAppReport`), also written as
`<checkpoint>/<session>.wfminiapi.report.json`:

| Field | Meaning |
|---|---|
| `telemetry_complete` | every event emitted and seen was written to the checkpoint; the checkpoint exists; telemetry was not stopped early |
| `tasks` | from AsyncFlow's events, tasks only (blocks excluded): `created`, `completed`, `failed`, `canceled`, `abandoned` (submitted, no end event, e.g. cancelled at shutdown), `not_submitted` (e.g. a failed dependency) |
| `ended_by` | how the block ended: `normal`; `exception` (any exception other than a cancel); `cancelled` (cancelled from outside: `asyncio.timeout`, a `TaskGroup`, `asyncio.run`, a supervisor); `engine_shutdown` (the engine was shut down under the block: a signal, an engine error, or a `flow.shutdown()` call that cancelled what the workflow awaited). A workflow that catches the cancel and returns ends `normal`, even after a signal. |
| `reasons` | why telemetry is incomplete |
| `exit_hint` | `0` if complete, `3` if not. wfMiniAPI never exits by itself; a script may do `sys.exit(app.report.exit_hint)`. |

## What the telemetry covers

| Recorded | By |
|---|---|
| The run: start, end, how it ended, versions, backends | `MiniApp` (`wfminiapi.RunStarted` / `RunEnded`) |
| Each task's lifecycle and outcome; workflow scopes and blocks; spans and resource metrics | AsyncFlow and RHAPSODY telemetry, as configured in `start_telemetry()` |
| Whether all of the above reached the checkpoint | `MiniApp` (`app.report`) |

**Nothing inside a task is recorded.** wfMiniAPI does not time kernels or steps, does
not pass a task context into tasks and adds no hooks on the backend. In-task telemetry
is the workflow developer's responsibility.

## Preflight

`require_dragon()` is optional. It raises `BackendUnavailableError` with a launch hint
when Dragon is not installed or the process was not started by the `dragon` launcher.
It never starts a Dragon runtime.

::: wfMiniAPI.api.app.MiniApp

::: wfMiniAPI.telemetry.verify.MiniAppReport

::: wfMiniAPI.telemetry.verify.verify_checkpoint

::: wfMiniAPI.api.preflight.require_dragon
