# Runs of many engines

In the [MiniApp script](api/miniapp.md#the-script), the workflow is one engine's tasks. A run can be more than that. It
can use other engines (one per population of agents, say), each in its own process, and
components that are not engines but leave records of their own. All of it is the run's
**composition** (`app.composition`), declared on the run's `MiniApp`. Its telemetry records
the composition and verifies it. `MiniApp` still builds and runs nothing: the script starts
every engine and component, as it does with one engine.

## What a run is made of

| | What it is | How it is declared | What it leaves | Who verifies it |
|---|---|---|---|---|
| **The run's engine** | the AsyncFlow engine whose `MiniApp` hands out the parts; its report is the run's report | `MiniApp(flow, telemetry)` | its telemetry checkpoint | its `MiniApp`, as with one engine |
| **A part** | each other engine of the run, in any process | `ref = app.part(name)` on the run's `MiniApp`; then, where that engine runs, `MiniApp(flow, telemetry, part_of=ref)` | its checkpoint, and its report in the run's parts directory | its own `MiniApp`; the run requires its report |
| **A component** | anything else that leaves records: an agent library's telemetry export, a model server's counters, the application's results | `app.component(name, files, verify=...)` (a `Component`) | its files | the verifier given, run by the run's `verify()` |
| **A check** | a function over the whole run: are the results right? | `app.check(name, verify=...)`, or `ok=` for a result computed in the run | | the run's `verify()`, last |
| **The parameters** | the run's configuration | `MiniApp(..., parameters=cfg)` or `app.parameters(cfg)` | a `wfminiapi.Parameters` event | |

How they fit together, with one part:

```text
the run's process                                  the part's process (any node)
-----------------                                  -----------------------------
engine E0 and its telemetry                        engine E1 and its telemetry
app = MiniApp(E0, telemetry, parameters=cfg)
  ref = app.part("p1")  ---- PartRef, picklable ---> part = MiniApp(E1, telemetry, part_of=ref)
                                                       ... E1 runs its tasks ...
                                                     part.verify() writes its report into
                                                       the run's parts directory
  app.component("agents", files, verify=...)
  app.check("results_match", verify=...)
app.verify(), after both engines are shut down:
  1. E0's checkpoint, as with one engine
  2. each part's report, from the parts directory
  3. each component's verifier, over its files
  4. each check, over all of the above (a Run)
  -> report.telemetry_complete (1 to 3) and report.checks_pass (4)
```

In the script, the run's `MiniApp` declares the composition inside its block:

```python
app = MiniApp(flow, telemetry, parameters=cfg)  # the run's engine and configuration
try:
    async with app, flow.workflow_scope("screening"):  # a scope name, as "motif_4" in the MiniApp script
        refs = [app.part(f"p{i}") for i in range(2)]  # one part per other engine
        await start_engines(refs)  # your code: each of its processes builds an engine
                                   # and MiniApp(flow, telemetry, part_of=ref)
        app.component("agents", run_dir / "otel", verify="mylib.telemetry:verify_export")
        app.check("simulations_match", verify="myapp.checks:simulations_match")
finally:
    ...  # the same nested finally
```

and each part, in whatever process it runs, has the same shape as a run of one engine:

```python
app = MiniApp(flow, telemetry, part_of=ref)  # same script shape and nested finally
```

`start_engines` stands for the application's own code: `MiniApp` does not start processes
or engines. The scope name, `"screening"` here and `"motif_4"` in the MiniApp script, only labels the
block's tasks in telemetry.

## What the run's verify() requires

- **Parts.** Each part's `verify()` writes its report into the run's parts directory
  (`<checkpoint>/<run_id>.parts/<name>.<session>.json`, an absolute path taken when the
  run's `MiniApp` attached). Every part's process must reach that path: with parts on
  other nodes, keep the run's checkpoint on a shared file system. Reports are written
  whole (a temporary file, then a rename), so the run never reads half of one. The run's
  `verify()` requires exactly one complete report from every part it handed out: a part that never reported, reported incomplete
  telemetry, was reported by two engines, or belongs to another run makes the run's
  telemetry incomplete (`part <name>: …`). Give each engine its own part; part names use
  letters, digits, `_` and `-`. The run's report lists each part's report under `parts`,
  and a part whose own checks failed adds a failed check `part <name>` to the run's.
- **Components.** A component is its name, its files and its verifier (a `Component`).
  The verifier takes the files and returns a `ComponentReport` (or a bool). An
  incomplete component makes the run's telemetry incomplete (`component <name>: …`);
  its `counts` are there for checks.
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
