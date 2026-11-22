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
| `run_id`, `part`, `parameters`, `completed_by_executable` | the run this engine belongs to, its part name, the run's configuration, completed tasks by function name |
| `parts`, `components`, `checks`, `checks_pass` | a run of many engines and components: see below |
| `exit_hint` | `0` if complete and every check passes, `3` if telemetry is incomplete, `4` if it is complete but a check failed. wfMiniAPI never exits by itself; a script may do `sys.exit(app.report.exit_hint)`. |

## A run of many engines and components

A run can use more than one engine (an engine per population of agents, for example),
each in its own process, and components besides the engines that record their own
telemetry (an agent library's OpenTelemetry export, the application's results). The
run's `MiniApp` assembles them:

```python
app = MiniApp(flow, telemetry)               # the run's engine, as above
try:
    async with app, flow.workflow_scope("campaign"):
        app.parameters(cfg)                   # the run's configuration
        ref = app.part("p0")                  # picklable: pass it to any process
        await run_population(ref, ...)        # builds its engine and its own MiniApp
        app.component("agents", run_dir / "otel", verify="mylib.telemetry:verify_export")
        app.check("simulations_match", verify="myapp.checks:simulations_match")
finally:
    ...                                       # the same nested finally
```

and, in each part, whatever process it runs in:

```python
app = MiniApp(flow, telemetry, part_of=ref)  # same script shape and nested finally
```

- **Parts.** Each part's `verify()` writes its report into the run's parts directory
  (`<checkpoint>/<run_id>.parts/<name>.<session>.json`, an absolute path taken when the
  run's `MiniApp` attached). Every part's process must reach that path: with parts on
  other nodes, keep the run's checkpoint on a shared file system. Reports are written
  whole (a temporary file, then a rename), so the run never reads half of one. The run's
  `verify()` requires exactly one complete report
  from every part it handed out: a part that never reported, reported incomplete
  telemetry, was reported by two engines, or belongs to another run makes the run's
  telemetry incomplete (`part <name>: …`). Give each engine its own part; part names use
  letters, digits, `_` and `-`. The run's report lists each part's report under `parts`,
  and a part whose own checks failed adds a failed check `part <name>` to the run's.
- **Components.** A verifier takes the component's files and returns a
  `ComponentReport` (or a bool). An incomplete component makes the run's telemetry
  incomplete (`component <name>: …`); its `counts` are there for checks.
- **Checks.** A check takes a `Run` (parameters, parts, components, the run's own
  report) and returns a bool or `(bool, detail)`; `app.check(name, ok=..., detail=...)`
  records a result computed in the run. Failed checks set `checks_pass` to `False` and
  `exit_hint` to `4`: they never make telemetry incomplete. Details, counts and
  parameters are written as plain JSON: keys become strings, sets and tuples lists,
  anything else its `str()`.
- **Offline.** The run's `verify()` also writes `<session>.wfminiapi.manifest.json`, with
  paths relative to its directory, so the run directory can be moved.
  `python -m wfMiniAPI.verify <checkpoint dir>` verifies the run again from its files;
  verifiers and checks named as `"module:function"` run again, the others keep the
  verdict recorded in the run. When the directory holds several runs, pass
  `--session <run engine's session id>`. A run of one engine with none of the above
  writes no manifest and sees no change.

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

::: wfMiniAPI.telemetry.run.PartRef

::: wfMiniAPI.telemetry.run.ComponentReport

::: wfMiniAPI.telemetry.run.Run

::: wfMiniAPI.telemetry.run.verify_run

::: wfMiniAPI.api.preflight.require_dragon
