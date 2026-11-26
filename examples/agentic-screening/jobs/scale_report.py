"""A scale run's report, for `jobs/scale` (jobs/README.md).

    write   RUN [--agents N] [--populations P] [--workers W] [--name NAME]
                                                           read the run's files, write
                                                           RUN/scale-report.json, and print it
    show    RUN                                            print RUN/scale-report.json, or the
                                                           headline of RUN/summary.json
    compare RUN_A RUN_B                                    the agents both runs have: the same
                                                           simulations, model calls and report?

`write` needs no Slurm and no framework, only the run's files:

- **every agent verifies.** It verifies each agent's report as the agent's row recorded it
  (`app/agents-p*.jsonl`), with the benchmark's own verification and the run's knobs
  (`campaign.json`), so it works when the mini-app wrote no summary. With `--agents`, the rung's
  N, it verifies agents 0 to N-1, and a run of another size fails. It counts duplicate rows and rows
  of agents outside the run, and names the populations that wrote no rows (`--populations`, else
  `campaign.json`'s). Where the mini-app wrote its checks (`app/checks.json`, `summary.json`), it
  gives them too.
- **where the time went.** The chain's jobs, from the accounting the report job keeps
  (`jobs.txt`): how long each waited in the queue and ran. The campaign job's own time, from its
  `timeline`: Dragon's and the services' start-up before the first agent, the agents, and the
  wind-down after the last. The agents' span against the simulations' bound: the seconds the
  simulations asked for over the workers, which no run can beat. And the agents' own time: in their
  simulations, in model calls (admission, then serving), and the rest, mostly waiting for a worker.
- **what the framework counted.** Populations write their rows when they end. When one was cut off
  (its timeout, the job's limit), the simulations and model calls come from the populations'
  telemetry (`otel/p*/metrics.jsonl`, the last whole export of each): with the agents by state at
  the end in the populations cut off, and the agents' time by state. A population that ended counts
  no agents as it shuts down, so its rows say where its agents were. With a model, it also gives a
  request's wait for admission and its time at the server.

`compare` checks two runs agent by agent, on the agents both have: with the same seed, agent i is
the same candidate at every size, so with the stub model its simulations, model calls and report
are the same on one node as on several.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1]
VERIFY_CHECK = "every_agent_finishes_and_verifies"


def read_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _number(x) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def well_formed(row) -> bool:
    """A row the report can use: the agent's id, its simulations, whether it finished, and a report
    with its numbers, or none."""
    if not isinstance(row, dict) or not isinstance(row.get("cid"), str) or "finished" not in row:
        return False
    sims, seconds, report = row.get("sims"), row.get("sim_seconds") or [], row.get("report")
    return (
        isinstance(sims, list)
        and isinstance(seconds, list)
        and all(isinstance(p, list) and len(p) == 2 and all(map(_number, p)) for p in seconds)
        and all(row.get(k) is None or _number(row[k]) for k in ("t_start", "t_end"))
        and (
            report is None
            or isinstance(report, dict)
            and all(_number(report.get(k)) for k in ("temperature", "doping", "score"))
        )
    )


def read_rows(run: Path) -> tuple[list[dict], int]:
    """The agents' rows, one file per population, and the lines that could not be read or used (a
    population killed while it wrote)."""
    rows, unreadable = [], 0
    for path in sorted((run / "app").glob("agents-p*.jsonl")):
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    row = None
                if well_formed(row):
                    rows.append(row)
                else:
                    unreadable += 1
    return rows, unreadable


def populations_without_rows(run: Path, populations: int | None) -> list[str]:
    """The run's populations (`populations`, else `campaign.json`'s) that wrote no rows."""
    if populations is None:
        config = (read_json(run / "campaign.json") or {}).get("config") or {}
        populations = config.get("populations")
    if not isinstance(populations, int):
        return []
    wrote = {p.name[len("agents-") : -len(".jsonl")] for p in (run / "app").glob("agents-p*.jsonl")}
    return [f"p{i:03d}" for i in range(populations) if f"p{i:03d}" not in wrote]


def read_timeline(run: Path) -> dict[str, float]:
    """The campaign job's marks (`campaign.sh`): job_start, dragon_start, dragon_end, job_end."""
    marks: dict[str, float] = {}
    try:
        lines = (run / "timeline").read_text().splitlines()
    except OSError:
        return marks
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            try:
                marks.setdefault(parts[0], float(parts[1]))  # a requeued job: its first start
                if parts[0].endswith("_end"):
                    marks[parts[0]] = float(parts[1])  # and its last end
            except ValueError:
                pass
    return marks


def _when(text: str) -> datetime.datetime | None:
    try:
        return datetime.datetime.fromisoformat(text)
    except ValueError:
        return None  # Unknown, None: not started


def read_jobs(run: Path, roles: str = "") -> list[dict]:
    """The chain's jobs from Slurm's accounting (`sacct -X -P`, kept by the report job), each with
    its role in the chain when `roles` names it ("test=101,model=,campaign=103")."""
    role = {i: r for r, _, i in (p.partition("=") for p in roles.split(",")) if i}
    try:
        lines = (run / "jobs.txt").read_text().splitlines()
    except OSError:
        return []
    if not lines:
        return []
    head = lines[0].split("|")
    jobs = []
    for line in lines[1:]:
        job = dict(zip(head, line.split("|"), strict=False))
        if not job.get("JobID"):
            continue
        submit, start = _when(job.get("Submit", "")), _when(job.get("Start", ""))
        job["wait_s"] = (start - submit).total_seconds() if submit and start else None
        job["role"] = role.get(job["JobID"].split(";")[0], job.get("JobName"))
        jobs.append(job)
    return jobs


def _points(export: dict) -> dict[str, list[dict]]:
    """One telemetry export's data points, by metric name."""
    points: dict[str, list[dict]] = collections.defaultdict(list)
    for rm in export.get("resource_metrics") or []:
        for sm in rm.get("scope_metrics") or []:
            for m in sm.get("metrics") or []:
                data = m.get("data") or {}
                points[m.get("name")] += [p for p in data.get("data_points") or [] if p]
    return points


def _by_state(points: list[dict]) -> dict[str, float]:
    out: dict[str, float] = collections.Counter()
    for p in points:
        state, value = (p.get("attributes") or {}).get("flowgentic.agent.state"), p.get("value")
        if state and _number(value):
            out[state] += value
    return dict(out)


def _counted(points: list[dict], **attrs) -> tuple[int, float]:
    """The count and sum of a histogram's points whose attributes match."""
    count, total = 0, 0.0
    for p in points:
        have = p.get("attributes") or {}
        if all(have.get(k) == v for k, v in attrs.items()) and _number(p.get("count")):
            count += p["count"]
            total += p.get("sum") or 0.0
    return count, total


def _last_export(path: Path) -> dict | None:
    """The last export of a metrics file that reads whole: a population killed while it wrote
    leaves a broken last line."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            export = json.loads(line)
        except ValueError:
            continue
        if isinstance(export, dict):
            return export
    return None


def read_telemetry(run: Path) -> dict | None:
    """What Flowgentic counted while the populations ran (`otel/p*/metrics.jsonl`; its counts are
    cumulative, so the last export of each population holds them): each population's agents by
    state at the end, the simulations that ran, the model calls served, a request's mean wait for
    admission and time at the server, the agents' seconds by state, and the seconds the exports
    cover. None without. A population that ended counts no agents in any state in its last export,
    made as it shut down: it is in `populations_ended`, not in `agents`."""
    agents: dict[str, dict[str, int]] = {}
    ended = []
    seconds: dict[str, float] = collections.Counter()
    sims = calls = admitted = served = 0
    admission_s = request_s = 0.0
    covers = 0.0
    populations = []
    for path in sorted((run / "otel").glob("p*/metrics.jsonl")):
        export = _last_export(path)
        if export is None:
            continue
        pop = path.parent.name
        populations.append(pop)
        points = _points(export)
        states = {k: int(v) for k, v in _by_state(points["flowgentic.agents"]).items()}
        if any(states.values()):
            agents[pop] = states
        elif states:
            ended.append(pop)
        seconds.update(_by_state(points["flowgentic.agent.time"]))
        sims += _counted(points["flowgentic.tool.duration"], **{
            "gen_ai.tool.name": "simulate", "flowgentic.tool.outcome": "ok"})[0]  # fmt: skip
        n, _ = _counted(points["flowgentic.fleet.call.duration"])
        calls += n
        n, total = _counted(points["flowgentic.fleet.admission.wait"])
        admitted, admission_s = admitted + n, admission_s + total
        n, total = _counted(points["flowgentic.fleet.request.duration"])
        served, request_s = served + n, request_s + total
        for p in points["flowgentic.fleet.call.duration"]:
            t0, t1 = p.get("start_time_unix_nano"), p.get("time_unix_nano")
            if _number(t0) and _number(t1):
                covers = max(covers, (t1 - t0) / 1e9)
    if not populations:
        return None
    return {
        "populations": populations,
        "agents": agents,
        "populations_ended": ended,
        "simulations": sims,
        "model_calls": calls,
        "admission_mean_s": round(admission_s / admitted, 1) if admitted else None,
        "request_mean_s": round(request_s / served, 2) if served else None,
        "agent_s_by_state": {k: round(v, 1) for k, v in seconds.items()},
        "covers_s": round(covers, 1) if covers else None,
    }


def _screening() -> None:
    """The benchmark's package, beside this directory: standard library only."""
    if str(EXAMPLE) not in sys.path:
        sys.path.insert(0, str(EXAMPLE))


def run_agents(run: Path) -> int | None:
    """The agents the run was started with (`campaign.json`), if it got that far."""
    knobs = ((read_json(run / "campaign.json") or {}).get("knobs")) or {}
    return knobs.get("agents") if isinstance(knobs.get("agents"), int) else None


def benchmark(run: Path, agents: int | None):
    """The run's knobs (`campaign.json`), as the benchmark's `Benchmark`, for `agents` agents (the
    rung's) when given; None when neither says how many."""
    _screening()
    from screening.benchmark import Benchmark

    knobs = dict(((read_json(run / "campaign.json") or {}).get("knobs")) or {})
    if agents is not None:
        knobs["agents"] = agents
    if "agents" not in knobs:
        return None
    return Benchmark.from_dict(knobs)


def verify_rows(run: Path, rows: list[dict], agents: int | None) -> dict | None:
    """Every agent's report as its row recorded it, against its candidate's best; with the rows
    that repeat an agent and the rows of agents outside the run. None when nothing says how many
    agents the run has."""
    _screening()
    from screening import physics
    from screening.verify import verify

    bench = benchmark(run, agents)
    if bench is None:
        return None
    expected = [physics.cid_of(i) for i in range(bench.agents)]
    seen = collections.Counter(r["cid"] for r in rows)
    first: dict[str, dict] = {}
    for r in rows:
        first.setdefault(r["cid"], r)
    reports = {cid: (first.get(cid) or {}).get("report") for cid in expected}
    out = verify(bench, reports, list(first.values()))
    twice = {cid: n for cid, n in seen.items() if n > 1}
    outside = sorted(set(seen) - set(expected))
    out["duplicate_rows"] = sum(n - 1 for n in twice.values())
    out["duplicate_examples"] = sorted(twice)[:5]
    out["rows_outside_the_run"] = sum(seen[c] for c in outside)
    out["outside_examples"] = outside[:5]
    return out


def workers_of(run: Path, given: int | None) -> int | None:
    """The run's simulation workers: its populations times each one's workers."""
    config = (read_json(run / "campaign.json") or {}).get("config") or {}
    pops, each = config.get("populations"), (config.get("population") or {}).get("workers")
    return pops * each if pops and each else given


def nodes_of(run: Path) -> dict[str, list[int]]:
    """Each node's populations and agents, from the populations' stats."""
    nodes: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for path in sorted((run / "populations").glob("p*/population.json")):
        stats = read_json(path) or {}
        node = nodes[str(stats.get("node", "?"))]
        node[0] += 1
        node[1] += stats.get("agents") or 0
    return dict(nodes)


def _hms(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _n(x) -> str:
    """A count with thousands' commas; anything else as it is ("?" for nothing)."""
    return f"{x:,}" if isinstance(x, int) else ("?" if x is None else str(x))


def _s(x) -> str:
    """Seconds, or "?" for nothing."""
    return "?" if x is None else f"{x:.0f}"


def _largest(item: tuple[str, float]) -> tuple[float, str]:
    """Largest first, then by name."""
    return (-item[1], item[0])


def _pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.1f} %" if whole else "?"


def time_of(rows: list[dict], marks: dict[str, float], workers: int | None) -> dict:
    """Where the time went: the campaign job's phases, the agents' span against the simulations'
    bound, and the agents' own seconds."""
    starts = [r["t_start"] for r in rows if r.get("t_start")]
    ends = [r["t_end"] for r in rows if r.get("t_end")]
    first, last = (min(starts), max(ends)) if starts and ends else (None, None)
    asked = sum(a for r in rows for a, _ in r.get("sim_seconds") or [])
    taken = sum(t for r in rows for _, t in r.get("sim_seconds") or [])
    lived = sum(r["t_end"] - r["t_start"] for r in rows if r.get("t_start") and r.get("t_end"))
    wait = sum(r.get("model_wait_s") or 0.0 for r in rows)
    serve = sum(r.get("model_service_s") or 0.0 for r in rows)

    def between(a, b):
        return round(b - a, 1) if a is not None and b is not None else None

    span = between(first, last)
    bound = asked / workers if workers else None
    return {
        "job_s": between(marks.get("job_start"), marks.get("job_end")),
        "before_dragon_s": between(marks.get("job_start"), marks.get("dragon_start")),
        "startup_s": between(marks.get("dragon_start"), first),
        "agents_s": span,
        "winddown_s": between(last, marks.get("dragon_end")),
        "cleanup_s": between(marks.get("dragon_end"), marks.get("job_end")),
        "bound_s": round(bound, 1) if bound else None,
        "span_over_bound": round(span / bound, 3) if span and bound else None,
        "simulation_core_s": round(asked, 1),
        "workers": workers,
        "agent_s": round(lived, 1),
        "agent_s_simulations": round(taken, 1),
        "agent_s_model_admission": round(wait, 1),
        "agent_s_model_serving": round(serve, 1),
        "agent_s_rest": round(lived - taken - wait - serve, 1),
    }


def write(
    run: Path,
    agents: int | None,
    workers: int | None,
    name: str | None,
    roles: str = "",
    populations: int | None = None,
) -> dict:
    """The report: `agents` is the rung's N (else the run's own), `populations` its populations
    (else the run's own)."""
    rows, unreadable = read_rows(run)
    summary = read_json(run / "summary.json")
    checks = read_json(run / "app" / "checks.json")
    if checks is None and summary:
        checks = summary.get("checks")
    checks = checks or []
    failure = read_json(run / "app" / "failure.json")
    workers = workers_of(run, workers)
    v = verify_rows(run, rows, agents)
    started_with = run_agents(run)
    other_size = agents is not None and started_with is not None and started_with != agents
    missing = populations_without_rows(run, populations)
    own = next((c.get("detail") for c in checks if c.get("name") == VERIFY_CHECK), None)
    t = time_of(rows, read_timeline(run), workers)
    jobs = read_jobs(run, roles)
    nodes = nodes_of(run)
    telemetry = read_telemetry(run)
    from_telemetry = bool(telemetry) and (bool(missing) or not rows)
    if from_telemetry:  # some population was cut off before it wrote its rows
        calls, sims = telemetry["model_calls"], telemetry["simulations"]
    else:
        calls = sum(r.get("model_calls") or 0 for r in rows)
        sims = sum(len(r.get("sims") or []) for r in rows)
    every = (
        bool(v)
        and v["verified"] == v["agents"]
        and not v["duplicate_rows"]
        and not v["rows_outside_the_run"]
    )
    ok = every and bool(summary) and bool(summary.get("ok")) and not other_size and not missing
    failing = [c.get("name") for c in checks if not c.get("ok")]
    if summary:
        failing += [c.get("name") for c in summary.get("checks") or [] if not c.get("ok")]
    failing = sorted(set(filter(None, failing)))

    out = {
        "name": name or run.name,
        "ok": ok,
        "summary_written": bool(summary),
        "every_agent_verifies": every,
        "verification": v,
        "agents_expected": agents,
        "agents_started": started_with,
        "populations_without_rows": missing,
        "verification_by_the_run": own,
        "checks_failing": failing,
        "populations_not_ended": failure,
        "unreadable_rows": unreadable,
        "time": t,
        "jobs": jobs,
        "nodes": nodes,
        "simulations": sims,
        "model_calls": calls,
        "counts_from": "telemetry" if from_telemetry else "rows",
        "telemetry": telemetry,
        "autonmi": read_json(run / "autonmi.json"),
    }
    out["lines"] = describe(out)
    (run / "scale-report.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def describe(out: dict) -> list[str]:
    """The report as lines to print."""
    v, own, t = out["verification"], out["verification_by_the_run"], out["time"]
    title, failing = out["name"], out["checks_failing"]
    if out["ok"]:
        lines = [f"{title}: ok, every agent verifies and every check passes"]
    elif not out["summary_written"]:
        lines = [f"{title}: the mini-app wrote no summary.json; the agents are from their rows"]
    else:
        lines = [f"{title}: FAILED" + (f": {', '.join(failing)}" if failing else "")]
    expected, started = out.get("agents_expected"), out.get("agents_started")
    if expected is not None and started is not None and started != expected:
        lines.append(
            f"the run was started with {_n(started)} agents, but the rung has {_n(expected)}"
        )
    if v:
        text = f"agents: {_n(v['verified'])} of {_n(v['agents'])} verify, from their rows"
        if own:
            text += f"; the run's own check: {_n(own.get('verified'))} of {_n(own.get('agents'))}"
        lines.append(text)
        if v["reasons"]:
            lines.append(f"    not verified: {v['reasons']}; for example {v['examples'][:3]}")
        if v.get("duplicate_rows"):
            twice, examples = _n(v["duplicate_rows"]), v["duplicate_examples"]
            lines.append(f"    duplicate rows: {twice}, for example {examples}")
        if v.get("rows_outside_the_run"):
            lines.append(
                f"    rows of agents outside the run: {_n(v['rows_outside_the_run'])}, "
                f"for example {v['outside_examples']}"
            )
    else:
        lines.append("agents: nothing to verify against (no campaign.json, and no --agents)")
    missing = out.get("populations_without_rows")
    if missing:
        lines.append(f"populations that wrote no rows: {', '.join(missing)}")
    tel, from_telemetry = out.get("telemetry"), out.get("counts_from") == "telemetry"
    if from_telemetry:  # where the agents of the populations cut off were
        cut = missing or tel["populations"]
        states = collections.Counter()
        for pop in cut:
            states.update(tel["agents"].get(pop) or {})
        every = sum(states.values())
        done = states.pop("done", 0)
        text = "    their agents" if missing else "agents"
        text += f" at the end, from telemetry: {_n(done)} of {_n(every)} done"
        rest = ", ".join(f"{s} {_n(n)}" for s, n in sorted(states.items(), key=_largest) if n)
        unknown = [pop for pop in cut if pop not in tel["agents"]]
        lines.append(
            text
            + (f"; {rest}" if rest else "")
            + (f"; {', '.join(unknown)} counted none at the end" if unknown else "")
        )
    if out["unreadable_rows"]:
        lines.append(f"rows that could not be read: {out['unreadable_rows']}")
    if out["populations_not_ended"]:
        lines.append(f"populations that did not end: {out['populations_not_ended']}")
    lines.append("where the time went:")
    jobs = [
        f"{j.get('role')} {j.get('State')}, waited {_hms(j['wait_s'])}, ran {j.get('Elapsed')}"
        for j in out["jobs"]
    ]
    if jobs:
        lines.append("    jobs: " + "; ".join(jobs))
    if t.get("before_dragon_s") and t["before_dragon_s"] >= 5:
        lines.append(
            f"    before Dragon, {_s(t['before_dragon_s'])} s: the guard, and the wait for the "
            "model job's servers"
        )
    if t["job_s"] is not None:
        lines.append(
            f"    the campaign job, {_s(t['job_s'])} s: start-up {_s(t['startup_s'])} s "
            "(Dragon, the store, the services and the model servers), "
            f"agents {_s(t['agents_s'])} s, wind-down {_s(t['winddown_s'])} s "
            f"(reports, ranking, telemetry, Dragon's exit), clean-up {_s(t['cleanup_s'])} s"
        )
    if t["bound_s"]:
        lines.append(
            f"    agents' span {_s(t['agents_s'])} s, {t['span_over_bound']} times "
            f"the simulations' bound of {_s(t['bound_s'])} s "
            f"({t['simulation_core_s']:,.0f} core-s over {t['workers']} workers)"
        )
    if t["agent_s"]:
        a = t["agent_s"]
        admission, serving = t["agent_s_model_admission"], t["agent_s_model_serving"]
        lines.append(
            f"    agents' own time: simulations {_pct(t['agent_s_simulations'], a)}, "
            f"model calls {_pct(admission + serving, a)} "
            f"(admission {_pct(admission, a)}, serving {_pct(serving, a)}), "
            f"waiting for a worker and the rest {_pct(t['agent_s_rest'], a)}"
        )
    if from_telemetry:
        seconds = {s: x for s, x in tel["agent_s_by_state"].items() if s != "done"}
        whole = sum(seconds.values())
        if whole:
            shares = sorted(seconds.items(), key=_largest)
            lines.append(
                "    agents' own time, from telemetry: "
                + ", ".join(f"{s} {_pct(x, whole)}" for s, x in shares)
            )
    if tel and tel.get("admission_mean_s") is not None and tel.get("request_mean_s") is not None:
        lines.append(
            f"    model calls: a request waited {_s(tel['admission_mean_s'])} s on average for "
            f"admission, then took {tel['request_mean_s']:.1f} s"
        )
    text = f"    {out['simulations']:,} simulations, {out['model_calls']:,} model calls"
    if from_telemetry:
        text += " (from telemetry: populations without rows)"
        if tel.get("covers_s"):
            text += f", {out['model_calls'] / tel['covers_s']:.1f} calls/s over the "
            text += f"{_s(tel['covers_s'])} s it covers"
    elif t["agents_s"]:
        text += f", {out['model_calls'] / t['agents_s']:.1f} calls/s over the agents' span"
    lines.append(text)
    nodes = out["nodes"]
    if nodes:
        each = [f"{n} {p} populations ({a:,} agents)" for n, (p, a) in sorted(nodes.items())]
        lines.append(f"nodes: {len(nodes)}: " + ", ".join(each))
    pin = out.get("autonmi")
    if pin:  # the AUTONMI commit the run imported, against the mini-app's AUTONMI_COMMIT
        text = f"autonmi: {(pin.get('commit') or 'an unknown commit')[:12]}"
        if pin.get("matches_pin"):
            text += ", the pinned commit"
        else:
            text += f", NOT the pinned {str(pin.get('pin'))[:12]}"
            text += " (its files changed)" if pin.get("dirty") else ""
        lines.append(text)
    return lines


def show(run: Path) -> list[str]:
    saved = read_json(run / "scale-report.json")
    if saved:
        return saved["lines"]
    summary = read_json(run / "summary.json")
    if not summary:
        return [f"no report and no summary.json yet in {run}"]
    bad = [c["name"] for c in summary.get("checks") or [] if not c.get("ok")]
    return [
        f"ok={summary.get('ok')}, agents {summary.get('agents')}, "
        f"simulations {summary.get('simulations')}, span {summary.get('simulations_span_s')} s; "
        "no report yet (jobs/scale report writes one)",
        "checks: all pass" if not bad else f"checks failing: {', '.join(bad)}",
    ]


def _key(row: dict) -> dict:
    rep = row.get("report") or {}
    return {
        "sims": [(round(t, 3), round(d, 5)) for t, d, *_ in row.get("sims") or []],
        "model_calls": row.get("model_calls"),
        "report": (rep.get("temperature"), rep.get("doping"), round(rep.get("score", 0.0), 6))
        if rep
        else None,
        "finished": bool(row.get("finished")),
    }


def compare(first: Path, second: Path) -> dict:
    a = {r["cid"]: r for r in read_rows(first)[0]}
    b = {r["cid"]: r for r in read_rows(second)[0]}
    common = sorted(set(a) & set(b))
    differ = {
        cid: {"first": _key(a[cid]), "second": _key(b[cid])}
        for cid in common
        if _key(a[cid]) != _key(b[cid])
    }
    return {
        "agents_first": len(a),
        "agents_second": len(b),
        "common": len(common),
        "identical": len(common) - len(differ),
        "differ": len(differ),
        "examples": dict(list(differ.items())[:5]),
        "simulations_of_common": [
            sum(len(a[c].get("sims") or []) for c in common),
            sum(len(b[c].get("sims") or []) for c in common),
        ],
        "model_calls_of_common": [
            sum(a[c].get("model_calls") or 0 for c in common),
            sum(b[c].get("model_calls") or 0 for c in common),
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="A scale run's report (jobs/README.md).")
    sub = p.add_subparsers(dest="what", required=True)
    w = sub.add_parser("write", help="write RUN/scale-report.json and print it")
    w.add_argument("run", type=Path)
    w.add_argument("--agents", type=int, help="the rung's agents: a run of another size fails")
    w.add_argument("--populations", type=int, help="the rung's populations (else the run's own)")
    w.add_argument(
        "--workers", type=int, help="simulation workers in all, when the run has no campaign.json"
    )
    w.add_argument("--name", help="the run's name in the report (default: its directory's)")
    w.add_argument("--jobs", default="", help="the jobs by role: test=ID,model=ID,campaign=ID")
    s = sub.add_parser("show", help="print the report, or the summary's headline")
    s.add_argument("run", type=Path)
    c = sub.add_parser("compare", help="the agents both runs have, agent by agent")
    c.add_argument("first", type=Path)
    c.add_argument("second", type=Path)
    a = p.parse_args(argv)
    if a.what == "write":
        out = write(a.run, a.agents, a.workers, a.name, a.jobs, a.populations)
        print("\n".join(out["lines"]))
        print(f"wrote {a.run / 'scale-report.json'}")
        return 0 if out["ok"] else 1
    if a.what == "show":
        print("\n".join(show(a.run)))
        return 0
    out = compare(a.first, a.second)
    print(json.dumps(out, indent=1))
    return 0 if out["common"] and not out["differ"] else 1


if __name__ == "__main__":
    sys.exit(main())
