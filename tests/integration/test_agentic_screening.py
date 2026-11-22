"""The agentic screening example (``examples/agentic-screening/``): a mini-app of many AsyncFlow
engines, one per population of agents, whose agents run on the AUTONMI framework.

The framework is installed separately, so the tests that run the example skip without it. The
benchmark's own test below runs everywhere, and so does the contract check
(``tests/equivalence/contract_check.py``), which reads the example's source.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "agentic-screening"
SCRIPT = EXAMPLE / "miniapp_screening.py"


def _installed(*modules: str) -> bool:
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except ModuleNotFoundError:
        return False


needs_framework = pytest.mark.skipif(
    not _installed("autonmi.campaign", "flowgentic.telemetry"),
    reason="needs the AUTONMI framework (installed separately)",
)


def run(tmp_path, *args: str) -> tuple[int, dict]:
    """The example in a process of its own, as a user runs it: 30 agents in two populations."""
    run_dir = tmp_path / "run"
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--agents", "30", "--populations", "2", "--workers", "4",
            "--sim-median", "0.01", "--no-burn", "--stagger-interval", "0.01",
            "--run-dir", str(run_dir), *args,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )  # fmt: skip
    summary = run_dir / "summary.json"
    assert summary.exists(), proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc.returncode, json.loads(summary.read_text())


def test_the_benchmark_verifies_reports_at_each_best_and_refuses_one_far_from_it(monkeypatch):
    """The benchmark is seeded: every candidate's best is known, so a run verifies itself."""
    monkeypatch.syspath_prepend(str(EXAMPLE))  # the pure part: no framework needed
    from screening import physics
    from screening.benchmark import Benchmark
    from screening.verify import verify

    bench = Benchmark(agents=20)
    reports, rows = {}, []
    for i in range(bench.agents):
        c = physics.candidate(i, bench)
        t, d = physics.optimum(c)
        reports[c.cid] = {"temperature": t, "doping": d, "score": physics.score(c, t, d)}
        rows.append({"cid": c.cid, "sims": [(t, d, f"res/{c.cid}/0")], "finished": True})
    assert verify(bench, reports, rows)["verified"] == 20
    far = physics.candidate(3, bench)
    reports[far.cid] = {"temperature": 300.0, "doping": 0.0, "score": 0.0}
    result = verify(bench, reports, rows)
    assert result["verified"] == 19 and result["reasons"] == {"report below tolerance": 1}


@needs_framework
def test_the_example_verifies_and_verifies_again_offline(tmp_path):
    rc, summary = run(tmp_path, "--lookup-calls", "0")
    assert rc == 0, summary["checks"]
    assert summary["agents"] == 30 and summary["populations"] == 2
    assert summary["exit_hint"] == 0 and all(c["ok"] for c in summary["checks"])
    (report,) = (tmp_path / "run" / "launcher").glob("*.wfminiapi.report.json")
    report = json.loads(report.read_text())
    assert sorted(report["parts"]) == ["p000", "p001"]
    assert set(report["components"]) == {"flowgentic", "application"}
    again = subprocess.run(
        [sys.executable, "-m", "wfMiniAPI.verify", str(tmp_path / "run" / "launcher")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert again.returncode == 0, again.stdout[-2000:] + again.stderr[-2000:]


@needs_framework
def test_lookups_go_through_the_example_s_own_mcp_servers(tmp_path):
    """Its reference-data MCP servers run as `python -m screening.mcp_server`, from the example."""
    rc, summary = run(tmp_path)
    assert rc == 0 and summary["exit_hint"] == 0, summary["checks"]
    services = tmp_path / "run" / "services"
    assert len(list(services.glob("mcp-server-*.log"))) == 4
    done = sorted(services.glob("toolservice-?.json"))
    assert sum(json.loads(p.read_text())["received"] for p in done) == 30  # one lookup per agent
