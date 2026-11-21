# Telemetry: what a run records

Every mini-app run records **telemetry**: a timestamped record of what happened. It
covers which tasks were created, handed to the backend, finished or failed; how busy
each node was; and how the run ended. You get it in files you can read, plot or
forward, plus a short report that says whether the record is complete.

This page explains how it works, what you get by default, how to read it, and its
limits.

## How it works

Three pieces work together; you create the first two yourself:

| Piece | What it does |
|---|---|
| **AsyncFlow** (`flow`) | Runs your workflow. For every task it emits an event when the task is created, ready, handed to the backend and finished. |
| **RHAPSODY telemetry** (`telemetry`, from `flow.start_telemetry()`) | Collects those events, samples node resources (CPU, memory, GPU), keeps counters and traces, and writes everything to one file. |
| **`MiniApp`** | Checks that the telemetry belongs to your flow, records when the run started and how it ended, and at the end verifies that the file is complete. |

```python
backend   = await DragonExecutionBackend()
flow      = await WorkflowEngine.create(backend=backend)
telemetry = await flow.start_telemetry(checkpoint_path="telemetry")   # where the file goes
app = MiniApp(flow, telemetry)
try:
    async with app, flow.workflow_scope("motif_4"):
        await miniapp_workflow(cfg, flow, logger)
finally:
    try:
        await flow.shutdown()
    finally:
        await telemetry.stop()     # writes out everything still queued, closes the file
        app.verify()               # reads the file, writes the report
print(app.report.telemetry_complete, app.report.tasks)
```

The order matters: the engine shuts down first (so the last task events are emitted),
then telemetry stops (so everything is written), then `verify()` reads the finished
file. See [MiniApp](api/miniapp.md) for why the `finally` is nested.

**Nothing inside a task is recorded.** wfMiniAPI does not time kernels or steps inside
a task. If you need that, record it yourself in the task.

## What you get

After the run, the `checkpoint_path` directory holds two files:

| File | Content |
|---|---|
| `<session>.<number>.telemetry.jsonl` | The checkpoint: every event, the final metrics and every trace span, one JSON object per line. `<session>` is the engine's id, e.g. `asyncflow.session.95f83ce1`. |
| `<session>.wfminiapi.report.json` | MiniApp's report: is the record complete, and how many tasks ended how. |

Each line of the checkpoint has a `section`: `event`, `metric` or `span`.

### Events

Events are the timeline of the run. Every event has:

- `event_type`, `event_time` (when it happened, Unix seconds) and `emit_time` (when it
  was recorded);
- `session_id`, `task_id`, `node_id` and `backend`;
- `trace_id` and `span_id`, which link it to its trace span.

| Event | When | Extra fields |
|---|---|---|
| `SessionStarted` | telemetry starts | – |
| `wfminiapi.RunStarted` | the `async with app` block is entered | `wfminiapi_version`, `asyncflow_version`, `rhapsody_version` (with the git commit when known), `backends`, `run_id` (the run engine's session id), `part` (this engine's part name in a run of many engines, else empty) |
| `wfminiapi.Parameters` | `app.parameters(...)` is called, or the block is entered after it | `parameters`: the run's configuration, as JSON |
| `TaskCreated` | a task or block is called | attributes: `executable` (the function's name), `task_type`, `depends_on`, `asyncflow.workflow_id` (the scope name) |
| `asyncflow.TaskResolved` | all its dependencies have finished | same attributes |
| `TaskQueued` | it is ready to be handed to the backend | same attributes |
| `TaskSubmitted` | it is handed to the backend | same attributes |
| `TaskStarted` | the backend accepted it (see [Limits](#limits)) | same attributes |
| `TaskCompleted` | it finished successfully | `duration_seconds` |
| `TaskFailed` | it raised, or its process failed | `duration_seconds`, `error_type` |
| `TaskCanceled` | it was cancelled through its backend | `duration_seconds` |
| `ResourceUpdate` | every 5 s, for each node | `cpu_percent`, `memory_percent`, `gpu_percent`, `disk_read_bytes`, `disk_write_bytes`, `net_sent_bytes`, `net_recv_bytes`; one extra event per GPU (`gpu_id`) on GPU nodes |
| `wfminiapi.RunEnded` | the block is left | `ended_by` (see [the report](#the-report)), `exception_type`, `duration_s` (the block's duration) |
| `SessionEnded` | `telemetry.stop()` | `duration_seconds` (the whole telemetry session) |

CPU and memory values are percentages at the moment of the sample. Disk and network
values are bytes since the previous sample.

Blocks (`@flow.block`) also emit `TaskCreated` and friends, with a `task_id` starting
with `block.`. The report does not count them as tasks.

### Metrics

One line with `section: "metric"` holds the final value of each counter, written when
telemetry stops:

| Metric | Meaning |
|---|---|
| `tasks_submitted`, `tasks_started`, `tasks_completed`, `tasks_failed`, `tasks_canceled` | how many tasks reached each state |
| `tasks_running` | tasks started and not yet finished, when the snapshot was taken |
| `task_duration_seconds_count`, `task_duration_seconds_sum` | number of finished tasks and their total run time; divide for the mean |
| `node_cpu_utilization`, `node_memory_utilization` | the last node CPU and memory sample, in percent |
| `node_gpu_utilization` | the last GPU sample, on GPU nodes only |

A counter that never moved (for example `tasks_canceled`) may be absent from the line.

### Traces (spans)

Spans give the same run as a tree of timed intervals, the format used by tracing tools
such as Jaeger:

```
session
└── workflow          one per flow.workflow_scope(...)
    ├── block         one per @flow.block call
    │   └── task
    └── task          one per task
```

Each span has `start_ns`, `end_ns`, `duration_ms`, `status_code`, `status_description`
and `parent_span_id`. A task span's attributes are `task_id`, `executable`,
`task_type`, `status` (`completed`, `failed`, ...), `backend` and
`asyncflow.workflow_id`.

### The report

`app.report`, also written as `<session>.wfminiapi.report.json`:

| Field | Meaning |
|---|---|
| `telemetry_complete` | `true` if every event that was emitted reached the file, the file exists, and telemetry was stopped properly (after the engine, and it finished writing) |
| `ended_by` | how the block ended: `normal`; `exception` (your workflow raised); `cancelled` (a timeout or another task cancelled it); `engine_shutdown` (Ctrl-C, SIGTERM or an engine error stopped it) |
| `tasks` | `created`, `completed`, `failed`, `canceled`; `abandoned` (handed to the backend, never finished, e.g. still running at Ctrl-C); `not_submitted` (never handed over, e.g. a dependency failed) |
| `events` | MiniApp's own events in the file, by type |
| `reasons` | why the record is incomplete, if it is |
| `notes` | things worth knowing that do not make it incomplete |
| `run_id`, `part` | the run this engine belongs to, and its part name (empty for the run's own engine) |
| `parameters` | the run's configuration, from `app.parameters(...)` |
| `completed_by_executable` | completed tasks, by function name |
| `parts`, `components`, `checks`, `checks_pass` | for a run of many engines and components: see [MiniApp](api/miniapp.md#a-run-of-many-engines-and-components) |
| `exit_hint` | `0` if complete and every check passes, `3` if telemetry is incomplete, `4` if it is complete but a check failed; use `sys.exit(app.report.exit_hint)` if you want the exit code to reflect it |

An incomplete report does not mean the workflow failed: it means the *record* has a
gap. The common reasons are:

| Reason | What happened | What to do |
|---|---|---|
| `stopped_before_exit: …` | `telemetry.stop()` was called inside the block, while the engine still ran | Stop telemetry only in the `finally`, after `flow.shutdown()` |
| `telemetry still running: …` | `app.verify()` was called before `telemetry.stop()` | Call it last, in the inner `finally` |
| `telemetry did not finish writing its queue …` | `stop()` stalled or was interrupted | Check for a blocked event loop or a second Ctrl-C |
| `not attached: …` | `MiniApp` refused the telemetry (`AttachError`) | Follow the error message: pass the manager the latest `flow.start_telemetry()` returned, right after creating it |
| `<event>: emitted N, written M` | events were lost on the way to the file | Report it: this is a bug |
| `part <name>: …` | a part handed out with `app.part()` left no report, an incomplete one, or one of another run | Check that engine: it must build its `MiniApp` with `part_of=` and call `app.verify()` |
| `component <name>: …` | the component's verifier found its records incomplete, or raised | Follow the verifier's reason |

## Settings

`flow.start_telemetry()` takes these options; the defaults are what the examples use:

| Option | Default | Effect |
|---|---|---|
| `checkpoint_path` | none | Directory for the checkpoint file. **Required by `MiniApp`**: without a file there is nothing to verify. |
| `resource_poll_interval` | `5.0` | Seconds between resource samples. A 15 s run gets about two samples; lower it (e.g. `1.0`) for short runs. |
| `checkpoint_interval` | none | Seconds between periodic writes of metrics and spans. By default they are written once, at `stop()`; set it to get them over time, or to keep them if the process is killed. |
| `span_processors`, `metric_readers` | none | OpenTelemetry processors and readers, to send spans and metrics to an external system (Jaeger, Prometheus, Grafana) as well as to the file. |
| `resource` | from `OTEL_SERVICE_NAME` | The OpenTelemetry resource that labels the data. |

## Reading the data

**Plot a run** (task timeline, tasks on the backend over time, node CPU and memory, run
time per task kind):

```shell
pip install -e ".[plot]"
python tools/plot_run.py telemetry/ run.png
```

The directory must hold one run.

**In Python**, for example the mean run time per task kind:

```python
import json, glob, collections

(path,) = glob.glob("telemetry/*.telemetry.jsonl")
times = collections.defaultdict(list)
for line in open(path):
    rec = json.loads(line)
    if rec.get("section") == "event" and rec["event_type"] == "TaskCompleted":
        times[rec["attributes"]["executable"]].append(rec["duration_seconds"])
for name, values in times.items():
    print(f"{name}: {len(values)} tasks, mean {sum(values) / len(values):.2f} s")
```

**With `jq`**, for example every failed task and its error type:

```shell
jq -c 'select(.event_type == "TaskFailed") | {task_id, executable: .attributes.executable, error_type}' telemetry/*.telemetry.jsonl
```

**While the run is going**, `telemetry.subscribe(callback)` calls `callback(event)` for
every event as it is processed. Keep the callback short: it runs in the telemetry loop.

## Limits

What the record does not tell you today, and why:

| Limit | Effect | Status |
|---|---|---|
| "Started" means "handed to the backend" | `TaskStarted` is emitted when the backend accepts the task, not when it starts running, so queue wait and run time are not separated | RHAPSODY, UP-01 |
| Tasks cancelled at shutdown have no end event | After Ctrl-C, tasks that were running show up as `abandoned` | AsyncFlow #125 adds these events |
| `error_type` can be `unknown` | On Dragon, a failed task's error class is not always passed on; the reason is in the run's own output | RHAPSODY |
| Disk and network on Dragon | Were `0.0` in our single-node Dragon runs; do not rely on them there | RHAPSODY / Dragon, to be confirmed |
| GPU values | Only on nodes with a GPU the backend can see | by design |
| Metrics are a final snapshot | Without `checkpoint_interval`, counters are written once, at the end; events and spans carry the timeline | by design; set the option |
| Nothing inside tasks | No per-kernel or per-step timing | by design; record it in your task |
| Warnings hidden on Dragon | Under the `dragon` launcher, library warnings (such as MiniApp's "telemetry INCOMPLETE" line) are dropped unless your script configures logging, e.g. `logging.basicConfig(level=logging.INFO)`. The report file is not affected | Dragon, UP-21 |

## Troubleshooting

- **No report file.** `app.verify()` was not called; a warning says so when the process
  exits. Use the script shape at the top of this page.
- **`AttachError` on entry.** The message says what to change. The usual cause is
  passing a manager other than the one `flow.start_telemetry()` returned, or entering
  `MiniApp` after tasks were already submitted.
- **On Dragon, the process does not exit.** Something was not shut down: check that the
  `finally` calls `flow.shutdown()` and `telemetry.stop()`, and that a backend you kept
  with `flow.shutdown(skip_execution_backend=True)` is shut down too. After killing a
  run, use `dragon-cleanup`.
- **Two runs in one directory.** `tools/plot_run.py` needs one run per directory; give
  each run its own `checkpoint_path`.
