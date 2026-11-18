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

- `telemetry` is not the manager returned by this `flow`'s `start_telemetry()`;
- telemetry is not running, or has no checkpoint file (`checkpoint_path=`);
- tasks were already submitted (enter `MiniApp` right after `start_telemetry()`).

After an `AttachError`, `MiniApp` has not taken over anything: the engine and telemetry
are still running and remain yours to shut down (`await flow.shutdown()`,
`await telemetry.stop()`). On Dragon the process does not exit until both are.

Then it records `wfminiapi.RunStarted` (wfMiniAPI, AsyncFlow and RHAPSODY versions,
backend classes).

**On exit**, whether normal, on an exception or after a signal, it:

1. makes sure the engine has shut down. If the workflow already did it, it just waits
   for the shutdown to finish; on an exception it calls `flow.shutdown()` right away;
2. waits until every telemetry event so far has been processed;
3. records `wfminiapi.RunEnded`, then stops telemetry (after the engine; never before);
4. verifies the checkpoint file, and sets `app.report`.

It never raises on exit and never replaces the workflow's own exception. Signals are
handled by AsyncFlow, as before.

## The report

`app.report` (`MiniAppReport`), also written as
`<checkpoint>/<session>.wfminiapi.report.json`:

| Field | Meaning |
|---|---|
| `telemetry_complete` | every event emitted and seen was written to the checkpoint; the checkpoint exists; telemetry was not stopped early |
| `tasks` | from AsyncFlow's events, tasks only (blocks excluded): `created`, `completed`, `failed`, `canceled`, `abandoned` (submitted, no end event, e.g. cancelled at shutdown), `not_submitted` (e.g. a failed dependency) |
| `interrupted` | the block exited on a cancellation (signal) |
| `reasons` | why telemetry is incomplete |
| `exit_hint` | `0` if complete, `3` if not. wfMiniAPI never exits by itself; a script may do `sys.exit(app.report.exit_hint)`. |

## What the telemetry covers

| Recorded | By |
|---|---|
| The run: start, end, interruption, versions, backends | `MiniApp` (`wfminiapi.RunStarted` / `RunEnded`) |
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
