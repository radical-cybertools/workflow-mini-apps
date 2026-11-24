# Troubleshooting

This page lists the problems a mini-app run can show, what causes each one, and what
to change. Most of them are reported by `MiniApp`, either as an `AttachError` when the
`MiniApp` block starts or as a reason in the report.

## `AttachError` when the MiniApp block starts

`MiniApp` refuses to attach when it could not verify the run. The message starts with
one of these:

| Message starts with | Cause | Fix |
|---|---|---|
| `telemetry is None` | `MiniApp(flow, None)` | Pass the object `flow.start_telemetry(...)` returned. |
| `this flow has no telemetry` | `flow.start_telemetry()` was never called | Call it before creating `MiniApp`. |
| `telemetry session ... is not this flow's` | The telemetry belongs to another engine | Pass the manager returned by *this* flow's `start_telemetry()`. |
| `telemetry is not the manager this flow emits into` | A manager built by hand, or one from an earlier `start_telemetry()` call | Pass the object the latest `flow.start_telemetry()` returned. |
| `telemetry is not running` | Telemetry was stopped before the block | Start telemetry right before the `MiniApp` block and stop it only in the `finally`. |
| `telemetry has no checkpoint file to verify` | `start_telemetry()` without `checkpoint_path` | Use `flow.start_telemetry(checkpoint_path="telemetry")`. |
| `telemetry checkpoint not found` | The directory holds no checkpoint for this session, or more than one | Give each run its own `checkpoint_path` directory. |
| `tasks were already submitted before MiniApp attached` | The workflow started before the `MiniApp` block | Enter the `MiniApp` block right after `start_telemetry()`, and run all tasks inside it. |
| `telemetry dispatch loop did not respond` | Telemetry did not process `MiniApp`'s marker within 30 seconds | Check that nothing blocks the event loop; see [RHAPSODY telemetry](https://radical-cybertools.github.io/rhapsody/telemetry/). |
| `this AsyncFlow has no joinable shutdown` | An AsyncFlow older than the one wfMiniAPI pins | Reinstall: `pip install -e ".[workflow]"`. |
| `RHAPSODY telemetry lacks drain_on_stop` | A RHAPSODY older than the one wfMiniAPI pins | Reinstall: `pip install -e ".[workflow]"`. |

Your script's `finally` still runs after an `AttachError`: it shuts the engine down and
stops telemetry, and `app.verify()` returns a report whose reason starts with
`not attached:`. Whether the report file is written depends on the message; see
[No report file](#no-report-file).

## The report says incomplete

`telemetry_complete` is `false` and `reasons` says why. An incomplete record does not
mean the workflow failed.

| Reason starts with | What happened | What to do |
|---|---|---|
| `telemetry still running` | `app.verify()` was called before `telemetry.stop()` | Call `app.verify()` last, in the inner `finally` ([the script](api/miniapp.md#the-script)). |
| `stopped_before_exit` | `telemetry.stop()` was called inside the `MiniApp` block, while the engine still ran | Stop telemetry only in the `finally`, after `flow.shutdown()`. |
| `telemetry did not finish writing its queue` | `telemetry.stop()` stalled or was interrupted | Check for a blocked event loop or a kill during the stop. |
| `not attached:` | `MiniApp` raised `AttachError` | See [the table above](#attacherror-when-the-miniapp-block-starts). |
| `MiniApp was never entered` | `app.verify()` without an `async with app` block | Run the workflow inside the `MiniApp` block. |
| `expected one checkpoint` | The checkpoint file is missing, or the directory holds two runs of the same session | Give each run its own `checkpoint_path` directory. |
| `<event>: emitted N, written M` or `<event>: seen N, written M` | Events were lost between telemetry and the file | Please [open an issue](https://github.com/radical-cybertools/workflow-mini-apps/issues) with the checkpoint attached. |
| `part <name>:` | A part handed out with `app.part()` left no report, an incomplete one, two (two engines given one part), or one of another run | Check that engine: it must build its `MiniApp` with `part_of=` and call `app.verify()`; see [Runs of many engines](runs.md). |
| `component <name>:` | The component's verifier found its records incomplete, or raised | Follow the verifier's reason. |
| `run-end event failed` or `verification failed` | An unexpected error inside `MiniApp` | Please open an issue with the message. |

**A note, not a reason.** `RunEnded not in the checkpoint: the engine shutdown stopped
telemetry before MiniApp's exit` can appear only with AsyncFlow versions whose
`shutdown()` also stops telemetry, and only when that shutdown starts before the
`MiniApp` block ends (Ctrl-C, SIGTERM, an engine error). The record is still complete;
only `MiniApp`'s end event is missing.

## Tasks counted as `abandoned` or `not_submitted`

These counts describe the workflow, not the record:

- **`abandoned`**: the task was handed to the backend but never ended, usually because
  Ctrl-C, a signal or an error shut the engine down while it ran.
- **`not_submitted`**: the task was created but never handed to the backend, usually
  because one of its dependencies failed.

See [how tasks are counted](telemetry.md#how-tasks-are-counted).

## No report file

There are two causes:

- **`app.verify()` was not called.** A warning says so when the process exits:
  `verify() was never called; no report written`. Use the
  [script shape](api/miniapp.md#the-script), which calls it in the inner `finally`.
- **`MiniApp` did not attach, before it knew the checkpoint directory.** `app.verify()`
  still returns a report with a `not attached:` reason and logs it, but has no
  directory to write the file to. This happens for every
  [`AttachError`](#attacherror-when-the-miniapp-block-starts) except
  `telemetry checkpoint not found`, `tasks were already submitted before MiniApp attached`,
  `RHAPSODY telemetry lacks drain_on_stop` and `telemetry dispatch loop did not respond`.
  Fix the `AttachError`.

## The process does not exit (Dragon)

Something was not shut down. Check that the script's `finally` calls `flow.shutdown()`
and `telemetry.stop()`, and that a backend kept with
`flow.shutdown(skip_execution_backend=True)` is shut down too. After killing a run,
clean up as described in [Run on HPC](hpc.md#after-a-killed-run).

## No warnings or report line under the `dragon` launcher

The `dragon` launcher hides library log messages unless the script configures logging.
See [Run on HPC](hpc.md#logging).

## `tools/plot_run.py` refuses the directory

The plot tool needs exactly one checkpoint and one report in the directory
(`expected one checkpoint and one report`). Give each run its own `checkpoint_path`,
and call `app.verify()` so the report exists.
