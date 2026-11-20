# MiniApp

`MiniApp` is a workflow mini-app's single contact point. You build the RADICAL
components yourself, and they stay visible and swappable: the execution backend,
the AsyncFlow engine and AsyncFlow telemetry. You also shut them down. `MiniApp`
records the run and verifies its telemetry; it never shuts anything down.

```python
from rhapsody.backends import DragonExecutionBackend
from radical.asyncflow import WorkflowEngine
from wfMiniAPI.api import MiniApp

backend   = await DragonExecutionBackend()
flow      = await WorkflowEngine.create(backend=backend)
telemetry = await flow.start_telemetry(checkpoint_path="telemetry")
app = MiniApp(flow, telemetry)
try:
    async with app, flow.workflow_scope("motif_4"):
        await miniapp_workflow(cfg, flow, logger)      # the unchanged workflow
finally:
    try:
        await flow.shutdown()
    finally:
        await telemetry.stop()
        app.verify()
print(app.report.telemetry_complete, app.report.tasks)
```

The `finally` is nested on purpose. If the engine's shutdown fails (a backend raises),
or a second Ctrl-C arrives during it, telemetry is still stopped and the report is
still written; the shutdown's error then propagates. With a flat `finally`, a failed
shutdown would skip both, leave the checkpoint open, and on Dragon the process may not
exit.

`telemetry.stop()` after `flow.shutdown()` is safe either way: if AsyncFlow's
`shutdown()` already stopped telemetry (radical.asyncflow #125), it does nothing. Keep
the `telemetry` variable: after such a shutdown `flow.telemetry` is `None`.

## What it does

**On entry**, it raises `AttachError` with a fix-it message if:

- `telemetry` is not the manager this `flow` emits into: pass the object the latest
  `flow.start_telemetry()` returned, not a manager built by hand, a wrapper, or the
  result of an earlier call;
- telemetry is not running, or has no checkpoint file (`checkpoint_path=`);
- tasks were already submitted (enter `MiniApp` right after `start_telemetry()`).

The script's `finally` handles an `AttachError` like any other error: it shuts the
engine down, stops telemetry, and `app.verify()` returns a report that says MiniApp
did not attach.

Then it records `wfminiapi.RunStarted` (wfMiniAPI, AsyncFlow and RHAPSODY versions,
backend classes).

**On exit**, whether normal, on an exception, a cancel or a signal, it records how the
block ended (`ended_by`) and queues `wfminiapi.RunEnded`, if telemetry is still running.
It awaits nothing, so a cancel cannot interrupt it, and it never raises over the
workflow's own exception. It does not shut the engine down or stop telemetry: the
script does that next. `RunEnded` is therefore queued before the engine's shutdown, so
the shutdown's own events (cancelled tasks, `SessionEnded`) come after it, and its
`duration_s` covers the block, not the shutdown.

**`app.verify()`**, called after `flow.shutdown()` and `telemetry.stop()`, reads the
checkpoint, sets `app.report` and writes `<checkpoint>/<session>.wfminiapi.report.json`.
It never raises:

- called while telemetry still runs, it returns an incomplete report saying so, and
  counts nothing (the file is still buffered);
- after an `AttachError`, or if the block was never entered, the report says so;
- an error while reading the checkpoint becomes a reason in the report.

If a script never calls `app.verify()`, a WARNING is logged when the process exits:
"verify() was never called; no report written".

Signals are handled by AsyncFlow, as before: its handler starts the engine's shutdown
and cancels the awaited futures, the block ends with `ended_by` `engine_shutdown`, and
the script's `await flow.shutdown()` waits for that same shutdown.

**Time limits.** `MiniApp` sets none, and its exit waits for nothing. An
`asyncio.timeout()` around the block bounds the workflow; the script's `finally`
(engine shutdown, telemetry stop) runs after it, and Dragon's backend shutdown blocks
the event loop while it runs. For a hard limit use a process-level one (the batch
walltime, `timeout -k`); after a kill on Dragon, run `dragon-cleanup` and check for
orphaned `_dragon_native_python_process_main` processes.

## The report

`app.report` (`MiniAppReport`, set by `app.verify()`), also written as
`<checkpoint>/<session>.wfminiapi.report.json`:

| Field | Meaning |
|---|---|
| `telemetry_complete` | every event emitted and seen was written to the checkpoint; the checkpoint exists; telemetry finished writing its queue; it was not stopped by hand while the engine still ran |
| `tasks` | from AsyncFlow's events, tasks only (blocks excluded): `created`, `completed`, `failed`, `canceled`, `abandoned` (submitted, no end event, e.g. cancelled at shutdown), `not_submitted` (e.g. a failed dependency) |
| `ended_by` | how the block ended: `normal`; `exception` (any exception other than a cancel); `cancelled` (cancelled from outside: `asyncio.timeout`, a `TaskGroup`, `asyncio.run`, a supervisor); `engine_shutdown` (the engine was shut down under the block: a signal, an engine error, or a `flow.shutdown()` call that cancelled what the workflow awaited). A workflow that catches the cancel and returns ends `normal`, even after a signal. |
| `reasons` | why telemetry is incomplete |
| `notes` | facts worth knowing that do not make it incomplete, e.g. "RunEnded not in the checkpoint: the engine shutdown stopped telemetry before MiniApp's exit" (Ctrl-C with an AsyncFlow whose `shutdown()` stops telemetry) |
| `exit_hint` | `0` if complete, `3` if not. wfMiniAPI never exits by itself; a script may do `sys.exit(app.report.exit_hint)`. |

## What the telemetry covers

The full list of events, metrics and spans, how to read them and their limits: [Telemetry](../telemetry.md).

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
