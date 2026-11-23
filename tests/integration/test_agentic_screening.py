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


# --- the knobs and the recipes -------------------------------------------------------------

# The README's recipes, as their knobs; each test runs one small (`run`: 12 agents in two
# populations, short simulations that sleep). The reasoning trade-off runs its cost spread and
# policy: its thinking budget and gain are not in this example yet.
RECIPES = {
    "cpu-heavy": (),
    "gpu-heavy": ("--plan-calls", "2", "--tool-result-tokens", "2000"),
    "model-traffic-heavy": ("--plan-calls", "4", "--lookup-calls", "3"),
    "reasoning-trade-off": ("--sim-cost-sigma", "0.5", "--think-policy", "proportional"),
    "no-gpus": (
        "--model", "stub", "--stub-latency", "0.001", "--stub-replicas", "2",
        "--stub-kv-tokens", "100000", "--stub-decode-rate", "30", "--stub-think-median", "320",
    ),
}  # fmt: skip


def script():
    """The example's script as a module, for its command line."""
    spec = importlib.util.spec_from_file_location("miniapp_screening", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_framework
def test_each_workload_flag_reaches_the_benchmark():
    m = script()
    cfg = m.parse(
        [
            "--agents", "7", "--seed", "3", "--max-sims", "5", "--sims-per-agent", "2",
            "--lookup-calls", "3", "--sim-distribution", "uniform", "--sim-median", "2.5",
            "--sim-sigma", "0.7", "--sim-range", "1", "4", "--sim-out-elems", "64", "--no-burn",
            "--mcp-latency-ms", "1", "2", "--no-suspend", "--sim-cost-sigma", "0.4",
            "--think-policy", "proportional", "--think-budget-max", "4096", "--think-gain-on",
            "asked", "--plan-calls", "2", "--tool-result-tokens", "500",
        ]
    )  # fmt: skip
    b = m.benchmark(cfg)
    assert (b.agents, b.seed, b.max_sims, b.sims_per_agent, b.lookup_calls) == (7, 3, 5, 2, 3)
    assert (b.sim_distribution, b.sim_median_s, b.sim_sigma) == ("uniform", 2.5, 0.7)
    assert b.sim_range_s == (1.0, 4.0) and b.sim_out_elems == 64 and b.sim_burn is False
    assert b.mcp_latency_ms == (1.0, 2.0) and b.suspend_fraction == 0.0
    assert b.sim_cost_sigma == 0.4 and (b.think_policy, b.think_budget_max) == (
        "proportional",
        4096,
    )
    assert b.think_gain_on == "asked" and (b.plan_calls, b.tool_result_tokens) == (2, 500)
    assert m.benchmark(m.parse([])) == m.Benchmark()  # every default is the knob's


@needs_framework
def test_the_system_flags_reach_the_campaign_and_their_short_names_work():
    m = script()
    cfg = m.parse(
        [
            "--cap", "32", "--cap-tokens", "1000", "--reply-reserve", "16", "--start-batch", "5",
            "--start-interval", "0.2", "--stub-replicas", "3", "--stub-kv-tokens", "50",
            "--managers", "2", "--manager-gib", "1.5", "--mcp-clients", "4",
            "--mcp-per-client", "8",
        ]
    )  # fmt: skip
    assert (cfg.cap_requests, cfg.cap_tokens, cfg.reply_reserve) == (32, 1000, 16)
    assert (cfg.stagger_batch, cfg.stagger_interval) == (5, 0.2)
    assert (cfg.stub_replicas, cfg.stub_kv_tokens) == (3, 50)
    assert (cfg.managers, cfg.manager_gib, cfg.mcp_clients, cfg.mcp_per_client) == (2, 1.5, 4, 8)


@pytest.mark.parametrize(
    "args, why",
    [
        (("--think-budget", "256"), "reasoning knob is not in this example yet"),
        (("--think-gain", "1"), "reasoning knob is not in this example yet"),
        (("--admission", "cache"), "cache-first admission is not in this example yet"),
    ],
)
@needs_framework
def test_knobs_not_in_this_example_yet_say_so(args, why):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 2 and why in proc.stderr, proc.stderr[-1000:]


@pytest.mark.parametrize("recipe", sorted(RECIPES))
@needs_framework
def test_each_recipe_runs_with_the_stub(tmp_path, recipe):
    rc, summary = run(tmp_path, "--agents", "12", *RECIPES[recipe])
    assert rc == 0 and summary["exit_hint"] == 0, summary["checks"]
    assert summary["agents"] == 12 and all(c["ok"] for c in summary["checks"])
