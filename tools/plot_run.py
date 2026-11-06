"""Plot one MiniApp run from its RHAPSODY telemetry checkpoint (task level only).

    pip install -e ".[plot]"
    python tools/plot_run.py <telemetry-dir> [out.png]

<telemetry-dir> is the checkpoint_path given to flow.start_telemetry(); it must hold one
run (one *.telemetry.jsonl and its *.wfminiapi.report.json).  "Running" is measured from
AsyncFlow's TaskStarted, which RHAPSODY emits when a task is handed to the backend, not
when it really starts (UP-01).
"""
import collections, glob, json, sys

if len(sys.argv) < 2:
    sys.exit(__doc__)
try:
    import matplotlib
except ImportError:
    sys.exit('matplotlib is missing: pip install -e ".[plot]"')
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tdir = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "run.png"
ckpts = glob.glob(f"{tdir}/*.telemetry.jsonl")
reports = glob.glob(f"{tdir}/*.wfminiapi.report.json")
if len(ckpts) != 1 or len(reports) != 1:
    sys.exit(f"{tdir}: expected one checkpoint and one report, found {len(ckpts)} and {len(reports)}")
with open(reports[0]) as fh:
    report = json.load(fh)
with open(ckpts[0]) as fh:
    events = [r for r in map(json.loads, fh) if r.get("section") == "event"]
t0 = min(e["event_time"] for e in events)

tasks = collections.defaultdict(dict)
for e in events:
    uid = e.get("task_id")
    if not uid or uid.startswith("block."):
        continue
    t = tasks[uid]
    t.setdefault("name", (e.get("attributes") or {}).get("executable", "?"))
    t[e["event_type"]] = e["event_time"] - t0
order = sorted(tasks, key=lambda u: tasks[u].get("TaskCreated", 0))
names = sorted({tasks[u]["name"] for u in order})
color = dict(zip(names, plt.cm.tab10.colors))

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
# 1. timeline: queue wait (light) + run (solid); failed tasks outlined in red
g = ax[0, 0]
for y, uid in enumerate(order):
    t = tasks[uid]
    sub, start = t.get("TaskSubmitted"), t.get("TaskStarted")
    end = t.get("TaskCompleted", t.get("TaskFailed", t.get("TaskCanceled")))
    if sub is not None and start is not None:
        g.barh(y, start - sub, left=sub, color=color[t["name"]], alpha=0.25)
    if start is not None and end is not None:
        g.barh(y, end - start, left=start, color=color[t["name"]],
               edgecolor="red" if "TaskFailed" in t else "none", linewidth=2)
    elif "TaskStarted" not in t:
        g.plot([t.get("TaskCreated", 0)], [y], "kx")  # never ran (e.g. failed dependency)
g.set(title="Task timeline (light: waiting, solid: handed to backend, red edge: failed, x: never ran)",
      xlabel="seconds since session start", ylabel="task (creation order)")
g.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=color[n]) for n in names], labels=names,
         loc="lower right", fontsize=8)

# 2. concurrency: running tasks over time
c = ax[0, 1]
edges = sorted([(t["TaskStarted"], 1) for t in tasks.values() if "TaskStarted" in t] +
               [(t.get("TaskCompleted", t.get("TaskFailed", t.get("TaskCanceled"))), -1)
                for t in tasks.values() if "TaskStarted" in t])
xs, ys, n = [0], [0], 0
for x, d in edges:
    if x is None:
        continue
    xs += [x, x]; ys += [n, n + d]; n += d
c.plot(xs, ys, drawstyle="steps-post")
c.set(title="Tasks handed to the backend over time", xlabel="seconds since session start", ylabel="tasks running")

# 3. node resources (RHAPSODY ResourceUpdate)
r = ax[1, 0]
ru = [e for e in events if e["event_type"] == "ResourceUpdate"]
if ru:
    r.plot([e["event_time"] - t0 for e in ru], [e["cpu_percent"] for e in ru], "o-", label="CPU %")
    r.plot([e["event_time"] - t0 for e in ru], [e["memory_percent"] for e in ru], "s-", label="memory %")
    r.legend()
r.set(title=f"Node resources ({len(ru)} samples; poll interval = start_telemetry resource_poll_interval)",
      xlabel="seconds since session start", ylabel="%")

# 4. run duration per task kind
b = ax[1, 1]
dur = collections.defaultdict(list)
for t in tasks.values():
    end = t.get("TaskCompleted", t.get("TaskFailed"))
    if "TaskStarted" in t and end is not None:
        dur[t["name"]].append(end - t["TaskStarted"])
b.boxplot([dur[n] for n in names if dur[n]], tick_labels=[n for n in names if dur[n]])
b.set(title="Run time per task kind", ylabel="seconds")

tc = report["tasks"]
fig.suptitle(f"{report['session_id']} - telemetry {'complete' if report['telemetry_complete'] else 'INCOMPLETE'} - "
             f"tasks: {tc['completed']} completed, {tc['failed']} failed, {tc['canceled']} canceled, "
             f"{tc['abandoned']} abandoned, {tc['not_submitted']} not submitted")
fig.tight_layout()
fig.savefig(out, dpi=110)
print("wrote", out)
