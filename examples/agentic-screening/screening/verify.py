"""The campaign's last step: read every report from the store, verify each agent, rank the candidates."""

from __future__ import annotations

from . import physics
from .benchmark import Benchmark


def read_reports(store, n: int) -> dict[str, dict | None]:
    return {physics.cid_of(i): store.get(f"report/{physics.cid_of(i)}", None) for i in range(n)}


def verify(bench: Benchmark, reports: dict[str, dict | None], rows: list[dict]) -> dict:
    """Every agent finished, its report is in the store and lies within tolerance of its candidate's
    best, and it ran 1 to `max_sims` simulations."""
    by_cid = {r["cid"]: r for r in rows}
    ok = 0
    reasons: dict[str, int] = {}
    examples: list[str] = []
    sims_hist: dict[int, int] = {}
    gap = []

    def fail(cid: str, why: str) -> None:
        reasons[why] = reasons.get(why, 0) + 1
        if len(examples) < 10:
            examples.append(f"{cid}: {why}")

    for cid, rep in reports.items():
        c = physics.candidate(physics.index_of(cid), bench)
        row = by_cid.get(cid)
        if row is None:
            fail(cid, "no agent record")
            continue
        n = len(row["sims"])
        sims_hist[n] = sims_hist.get(n, 0) + 1
        if row.get("error"):
            fail(cid, f"error: {row['error'][:80]}")
        elif not row["finished"]:
            fail(cid, "did not finish with a report")
        elif rep is None:
            fail(cid, "no report in the store")
        elif not 1 <= n <= bench.max_sims:
            fail(cid, f"{n} simulations")
        elif not physics.verifies(c, rep["temperature"], rep["doping"], bench):
            fail(cid, "report below tolerance")
        else:
            ok += 1
        if rep is not None:
            gap.append(1 - rep["score"] / c.best)
    gap.sort()
    return {
        "agents": len(reports),
        # With the gain on the thinking tokens used, each agent's simulations depend on the model and
        # are not known in advance: the run verifies the reports (within tolerance, 1 to max_sims).
        "simulations_known_in_advance": physics.counts_known(bench),
        "verified": ok,
        "failed": len(reports) - ok,
        "reasons": reasons,
        "examples": examples,
        "simulations_per_agent": dict(sorted(sims_hist.items())),
        "report_gap_to_best": {
            "median": round(gap[len(gap) // 2], 5) if gap else None,
            "max": round(gap[-1], 5) if gap else None,
        },
    }


# A simulation that ends within this of its seconds ran them: a timer's and a scheduler's slack.
SLACK_S = 0.01


def overruns(rows: list[dict], slack_s: float = SLACK_S) -> dict:
    """The simulations against their seconds, from the rows' `sim_seconds` ([asked, taken] each):
    how many overran them and the largest overrun, and how many ended before them. A simulation's
    seconds are its cost, not a deadline, so an overrun is reported, not failed; a simulation that
    ended early did not do its work."""
    pairs = [(asked, taken) for r in rows for asked, taken in r.get("sim_seconds", [])]
    over = [taken - asked for asked, taken in pairs if taken > asked + slack_s]
    return {
        "simulations": len(pairs),
        "overran": len(over),
        "largest_overrun_s": round(max(over), 3) if over else 0.0,
        "slack_s": slack_s,
        "ended_early": sum(1 for asked, taken in pairs if taken < asked - slack_s),
    }


def rank(reports: dict[str, dict | None], top: int = 10) -> list[dict]:
    found = [r for r in reports.values() if r is not None]
    found.sort(key=lambda r: r["score"], reverse=True)
    return [
        {
            "material": r["material"],
            "score": round(r["score"], 4),
            "temperature": r["temperature"],
            "doping": r["doping"],
        }
        for r in found[:top]
    ]
