"""A scale run's report, for `jobs/scale` (jobs/README.md).

    write   RUN [--agents N] [--workers W] [--name NAME]   read the run's files, write
                                                           RUN/scale-report.json, and print it
    show    RUN                                            print RUN/scale-report.json, or the
                                                           headline of RUN/summary.json
    compare RUN_A RUN_B                                    the agents both runs have: the same
                                                           simulations, model calls and report?

`write` needs no Slurm and no framework, only the run's files:

- **every agent verifies.** It verifies each agent's report as the agent's row recorded it
  (`app/agents-p*.jsonl`), with the benchmark's own verification and the run's knobs
  (`campaign.json`), so it works when the mini-app wrote no summary. Where the mini-app wrote its
  checks (`app/checks.json`, `summary.json`), it gives them too.
- **where the time went.** The chain's jobs, from the accounting the report job keeps
  (`jobs.txt`): how long each waited in the queue and ran. The campaign job's own time, from its
  `timeline`: Dragon's and the services' start-up before the first agent, the agents, and the
  wind-down after the last. The agents' span against the simulations' bound: the seconds the
  simulations asked for over the workers, which no run can beat. And the agents' own time: in their
  simulations, in model calls (admission, then serving), and the rest, mostly waiting for a worker.

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


def read_rows(run: Path) -> tuple[list[dict], int]:
    """The agents' rows, one file per population, and the lines that could not be read (a
    population killed while it wrote)."""
    rows, unreadable = [], 0
    for path in sorted((run / "app").glob("agents-p*.jsonl")):
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    unreadable += 1
    return rows, unreadable


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


def read_jobs(run: Path) -> list[dict]:
    """The chain's jobs from Slurm's accounting (`sacct -X -P`, kept by the report job)."""
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
        jobs.append(job)
    return jobs


def _screening() -> None:
    """The benchmark's package, beside this directory: standard library only."""
    if str(EXAMPLE) not in sys.path:
        sys.path.insert(0, str(EXAMPLE))


def benchmark(run: Path, agents: int | None):
    """The run's knobs (`campaign.json`), as the benchmark's `Benchmark`."""
    _screening()
    from screening.benchmark import Benchmark

    campaign = read_json(run / "campaign.json") or {}
    knobs = dict(campaign.get("knobs") or {})
    if agents is not None and "agents" not in knobs:
        knobs["agents"] = agents
    if "agents" not in knobs:
        sys.exit(f"{run}: no campaign.json, so give --agents")
    return Benchmark.from_dict(knobs)


def verify_rows(run: Path, rows: list[dict], agents: int | None) -> dict:
    """Every agent's report as its row recorded it, against its candidate's best."""
    _screening()
    from screening import physics
    from screening.verify import verify

    bench = benchmark(run, agents)
    by_cid = {r["cid"]: r for r in rows if "cid" in r}
    reports = {
        physics.cid_of(i): (by_cid.get(physics.cid_of(i)) or {}).get("report")
        for i in range(bench.agents)
    }
    return verify(bench, reports, rows)


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


def write(run: Path, agents: int | None, workers: int | None, name: str | None) -> dict:
    rows, unreadable = read_rows(run)
    summary = read_json(run / "summary.json")
    checks = read_json(run / "app" / "checks.json")
    if checks is None and summary:
        checks = summary.get("checks")
    checks = checks or []
    failure = read_json(run / "app" / "failure.json")
    workers = workers_of(run, workers)
    v = verify_rows(run, rows, agents) if rows or agents else None
    own = next((c.get("detail") for c in checks if c.get("name") == VERIFY_CHECK), None)
    t = time_of(rows, read_timeline(run), workers)
    jobs = read_jobs(run)
    nodes = nodes_of(run)
    calls = sum(r.get("model_calls") or 0 for r in rows)
    sims = sum(len(r.get("sims") or []) for r in rows)
    every = bool(v) and v["verified"] == v["agents"]
    ok = every and bool(summary) and bool(summary.get("ok"))
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
        "verification_by_the_run": own,
        "checks_failing": failing,
        "populations_not_ended": failure,
        "unreadable_rows": unreadable,
        "time": t,
        "jobs": jobs,
        "nodes": nodes,
        "simulations": sims,
        "model_calls": calls,
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
    if v:
        text = f"agents: {_n(v['verified'])} of {_n(v['agents'])} verify, from their rows"
        if own:
            text += f"; the run's own check: {_n(own.get('verified'))} of {_n(own.get('agents'))}"
        lines.append(text)
        if v["reasons"]:
            lines.append(f"    not verified: {v['reasons']}; for example {v['examples'][:3]}")
    if out["unreadable_rows"]:
        lines.append(f"rows that could not be read: {out['unreadable_rows']}")
    if out["populations_not_ended"]:
        lines.append(f"populations that did not end: {out['populations_not_ended']}")
    lines.append("where the time went:")
    jobs = [
        f"{j.get('JobName')} {j.get('State')}, waited {_hms(j['wait_s'])}, ran {j.get('Elapsed')}"
        for j in out["jobs"]
    ]
    if jobs:
        lines.append("    jobs: " + "; ".join(jobs))
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
    text = f"    {out['simulations']:,} simulations, {out['model_calls']:,} model calls"
    if t["agents_s"]:
        text += f", {out['model_calls'] / t['agents_s']:.1f} calls/s over the agents' span"
    lines.append(text)
    nodes = out["nodes"]
    if nodes:
        each = [f"{n} {p} populations ({a:,} agents)" for n, (p, a) in sorted(nodes.items())]
        lines.append(f"nodes: {len(nodes)}: " + ", ".join(each))
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
    w.add_argument("--agents", type=int, help="when the run has no campaign.json")
    w.add_argument(
        "--workers", type=int, help="simulation workers in all, when the run has no campaign.json"
    )
    w.add_argument("--name", help="the run's name in the report (default: its directory's)")
    s = sub.add_parser("show", help="print the report, or the summary's headline")
    s.add_argument("run", type=Path)
    c = sub.add_parser("compare", help="the agents both runs have, agent by agent")
    c.add_argument("first", type=Path)
    c.add_argument("second", type=Path)
    a = p.parse_args(argv)
    if a.what == "write":
        out = write(a.run, a.agents, a.workers, a.name)
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
