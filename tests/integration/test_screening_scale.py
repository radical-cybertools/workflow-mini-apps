"""The scale ladder of the agentic screening example (``examples/agentic-screening/jobs/``).

``jobs/scale`` submits a chain of Slurm jobs; these tests run it with ``dry-run``, or with fake
``sbatch``, ``scancel`` and ``sacct`` on ``PATH``, so they need no Slurm. The report
(``jobs/scale_report.py``) runs on run directories written here: no framework and no GPU.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "agentic-screening"
JOBS = EXAMPLE / "jobs"
SCALE = JOBS / "scale"
REPORT = JOBS / "scale_report.py"
RUNS = "/scratch/runs"

SITE = f"""
AUTONMI_VENV=/opt/venv
AUTONMI_RUNS={RUNS}
SLURM_ACCOUNT=acct
SLURM_CPU_PARTITION=cpu
SLURM_GPU_PARTITION=gpu
SLURM_GPU_GRES=gpu:a100:1
AUTONMI_NODE_MEM=300G
AUTONMI_DRAGON_ARGS=(-t tcp --network-prefix '^ib\\w*$')
"""

ONE_RUN = ("NODES", "GPUS", "TIME", "TIMEOUT", "MODEL_TIME", "ARGS", "TAG")


def site(tmp_path, text: str = SITE, **settings) -> Path:
    path = tmp_path / "site.env"
    path.write_text(text + "".join(f"{k}={v}\n" for k, v in settings.items()))
    return path


def scale(tmp_path, *args: str, env_file: Path | None = None, **env) -> subprocess.CompletedProcess:
    """jobs/scale as a user runs it, with only the settings given: none of the caller's."""
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("AUTONMI_", "SLURM_")) and k not in ONE_RUN
    }
    clean["AUTONMI_LOCAL_ENV"] = str(env_file or site(tmp_path))
    clean.update(env)
    return subprocess.run(
        ["bash", str(SCALE), *args], capture_output=True, text=True, env=clean, timeout=60
    )


def jobs_of(out: str) -> dict[str, str]:
    """The dry run's jobs: label -> the sbatch command."""
    return dict(re.findall(r"^<(\w+)>: sbatch (.*)$", out, re.M))


def option(cmd: str, name: str) -> str:
    """The value of one sbatch or mini-app option in a command line."""
    found = re.search(rf"(?:^| ){re.escape(name)}[= ](\S+)", cmd)
    assert found, f"{name} not in {cmd}"
    return found.group(1)


@pytest.mark.parametrize("path", sorted(p.name for p in JOBS.iterdir() if p.suffix in ("", ".sh")))
def test_each_job_script_parses(path):
    subprocess.run(["bash", "-n", str(JOBS / path)], check=True)


def test_a_dry_run_prints_the_chain_and_submits_nothing(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for tool in ("sbatch", "scancel"):  # any call fails the run
        (bin_ / tool).write_text("#!/bin/sh\necho called >&2\nexit 9\n")
        (bin_ / tool).chmod(0o755)
    r = scale(tmp_path, "dry-run", "10k", PATH=f"{bin_}:{os.environ['PATH']}")
    assert r.returncode == 0, r.stderr
    assert "called" not in r.stderr and r.stdout.rstrip().endswith("dry run: nothing submitted")
    jobs = jobs_of(r.stdout)
    assert list(jobs) == ["test", "model", "campaign", "report"]
    model, campaign, report = jobs["model"], jobs["campaign"], jobs["report"]
    assert option(model, "-d") == "afterok:<test>" and option(model, "--gres") == "gpu:a100:2"
    assert option(model, "-p") == "gpu" and model.endswith(f"{RUNS}/scale-10k-fleet 2")
    assert option(campaign, "-d") == "after:<model>" and option(campaign, "-N") == "2"
    assert "--exclusive" in campaign and option(campaign, "--cpus-per-task") == "96"
    assert option(campaign, "--populations") == "4" and option(campaign, "--workers") == "44"
    assert option(campaign, "--model") == "openai" and option(campaign, "--cap") == "96"
    assert option(campaign, "--endpoints-file") == f"{RUNS}/scale-10k-fleet/endpoints"
    assert (
        option(campaign, "--signal") == "B:USR1@240"
        and option(campaign, "-D") == f"{RUNS}/scale-10k"
    )
    assert option(report, "-d") == "afterany:<campaign>"
    assert report.endswith("test=<test>,model=<model>,campaign=<campaign>")
    assert all(option(cmd, "-A") == "acct" for cmd in jobs.values())
    assert all("--export=ALL" in cmd for cmd in jobs.values())


def test_with_the_stub_there_is_no_model_job(tmp_path):
    r = scale(tmp_path, "dry-run", "1k", "stub")
    assert r.returncode == 0, r.stderr
    jobs = jobs_of(r.stdout)
    assert list(jobs) == ["test", "campaign", "report"]
    campaign = jobs["campaign"]
    assert option(campaign, "-d") == "afterok:<test>" and option(campaign, "--model") == "stub"
    assert (
        f"campaign.sh {RUNS}/scale-1k-stub none " in campaign and "--endpoints-file" not in campaign
    )
    assert jobs["report"].endswith("test=<test>,model=,campaign=<campaign>")


@pytest.mark.parametrize(
    "rung, agents, nodes, gpus, cap, mcp, time, timeout, model_time",
    [
        ("1k", 1000, 1, 1, 96, 4, "00:40:00", 900, "02:00:00"),
        ("10k", 10000, 2, 2, 96, 4, "01:15:00", 2700, "02:00:00"),
        ("100k", 100000, 10, 2, 320, 4, "02:00:00", 4800, "08:00:00"),
        ("300k", 300000, 30, 8, 960, 8, "02:00:00", 4800, "08:00:00"),
    ],
)
def test_each_rung_has_the_table_s_shape_on_nodes_of_96_cpus(
    tmp_path, rung, agents, nodes, gpus, cap, mcp, time, timeout, model_time
):
    r = scale(tmp_path, "dry-run", rung)
    assert r.returncode == 0, r.stderr
    jobs = jobs_of(r.stdout)
    campaign, model = jobs["campaign"], jobs["model"]
    assert option(campaign, "--agents") == str(agents) and option(campaign, "-N") == str(nodes)
    assert option(campaign, "--populations") == str(2 * nodes)
    assert option(campaign, "--workers") == "44" and option(campaign, "--cap") == str(cap)
    assert option(campaign, "--mcp-servers") == str(mcp) and option(campaign, "-t") == time
    assert option(campaign, "--timeout") == str(timeout)
    assert option(model, "--gres") == f"gpu:a100:{gpus}" and option(model, "-t") == model_time
    assert option(model, "--cpus-per-task") == str(12 * gpus)
    assert f"{agents} agents on {nodes} CPU node(s) of 96 CPUs" in r.stdout


def test_the_benchmark_s_work_per_agent_is_the_ladder_s(monkeypatch):
    """The bound the dry run prints uses 31.45 core-seconds per agent: the benchmark's physics at
    its defaults, which the report measures from the rows."""
    monkeypatch.syspath_prepend(str(EXAMPLE))
    from screening import physics
    from screening.benchmark import Benchmark

    bench = Benchmark(agents=10_000)
    work = 0.0
    for i in range(bench.agents):
        c = physics.candidate(i, bench)
        t, d = physics.reference(c, bench)
        for n in range(1, bench.max_sims + 1):
            work += physics.sim_seconds(c, t, d, bench)
            _, done, nxt = physics.analysis(c, t, d, n, bench)
            if done:
                break
            t, d = nxt
    assert re.search(r"^CORE_CS_PER_AGENT=3145\b", SCALE.read_text(), re.M)
    assert work / bench.agents == pytest.approx(31.45, abs=0.01)


def test_the_workers_follow_the_site_s_cpus_and_the_limits_follow_the_workers(tmp_path):
    """On nodes of 64 CPUs, shared: 28 workers per population, and limits 88 / 56 times longer."""
    env_file = site(tmp_path, AUTONMI_NODE_CPUS=64, AUTONMI_NODE_EXCLUSIVE=0)
    r = scale(tmp_path, "dry-run", "10k", "stub", env_file=env_file)
    assert r.returncode == 0, r.stderr
    campaign = jobs_of(r.stdout)["campaign"]
    assert option(campaign, "--workers") == "28" and option(campaign, "--populations") == "4"
    assert "--exclusive" not in campaign and option(campaign, "--cpus-per-task") == "64"
    assert option(campaign, "-t") == "01:58:00" and option(campaign, "--timeout") == "4243"
    assert "112 in all" in r.stdout and "bound about 2808 s" in r.stdout
    r = scale(tmp_path, "dry-run", "10k", env_file=env_file)
    assert option(jobs_of(r.stdout)["model"], "-t") == "02:43:00"  # it waits for the campaign too


def test_more_cpus_give_more_workers_and_never_shorter_limits(tmp_path):
    env_file = site(tmp_path, AUTONMI_NODE_CPUS=192, AUTONMI_NODE_POPULATIONS=4)
    r = scale(tmp_path, "dry-run", "100k", env_file=env_file)
    assert r.returncode == 0, r.stderr
    campaign = jobs_of(r.stdout)["campaign"]
    assert option(campaign, "--workers") == "46" and option(campaign, "--populations") == "40"
    assert option(campaign, "-t") == "02:00:00" and option(campaign, "--timeout") == "4800"


def test_one_run_s_settings_change_only_that_run(tmp_path):
    r = scale(
        tmp_path, "dry-run", "10k", "stub",
        NODES="1", TAG="1n", TIME="00:30:00", TIMEOUT="600", ARGS="--span-ratio 1 --seed 3",
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    campaign = jobs_of(r.stdout)["campaign"]
    assert option(campaign, "-N") == "1" and option(campaign, "--populations") == "2"
    assert option(campaign, "-t") == "00:30:00" and option(campaign, "--timeout") == "600"
    assert campaign.endswith("--model stub --span-ratio 1 --seed 3")
    assert option(campaign, "-o") == f"{RUNS}/scale-10k-stub-1n/campaign-%j.out"
    r = scale(tmp_path, "dry-run", "10k", GPUS="1")
    assert option(jobs_of(r.stdout)["model"], "--gres") == "gpu:a100:1"


def test_the_site_s_own_options_and_model_settings_reach_the_jobs(tmp_path):
    env_file = site(
        tmp_path,
        AUTONMI_QOS_TEST="test",
        AUTONMI_QOS_CAMPAIGN="short",
        AUTONMI_QOS_MODEL="gpu-short",
        AUTONMI_CAP_TOKENS=90000,
        AUTONMI_MODEL_NAME="Nemotron-3-Nano",
        AUTONMI_CAMPAIGN_ARGS='"--span-ratio 1"',
        AUTONMI_SBATCH_CAMPAIGN='"--exclude=node3 --reservation=r1"',
        AUTONMI_SBATCH_MODEL="--constraint=b200",
        SLURM_GPU_GRES="",
        SLURM_ACCOUNT="",
    )
    r = scale(tmp_path, "dry-run", "10k", env_file=env_file)
    assert r.returncode == 0, r.stderr
    jobs = jobs_of(r.stdout)
    assert option(jobs["test"], "-q") == "test" and option(jobs["report"], "-q") == "test"
    assert option(jobs["campaign"], "-q") == "short" and option(jobs["model"], "-q") == "gpu-short"
    assert "--gres" not in jobs["model"] and all(" -A " not in c for c in jobs.values())
    assert option(jobs["campaign"], "--cap-tokens") == "90000"
    assert option(jobs["campaign"], "--model-name") == "Nemotron-3-Nano"
    assert jobs["campaign"].endswith("--span-ratio 1")
    assert option(jobs["campaign"], "--exclude") == "node3"
    assert option(jobs["campaign"], "--reservation") == "r1"
    assert option(jobs["model"], "--constraint") == "b200" and "--exclude" not in jobs["model"]


@pytest.mark.parametrize(
    "settings, args, why",
    [
        ({"AUTONMI_NODE_MEM": ""}, ("10k", "stub"), "AUTONMI_NODE_MEM is not set"),
        ({"AUTONMI_RUNS": "runs"}, ("10k", "stub"), "AUTONMI_RUNS must be an absolute path"),
        ({"SLURM_GPU_GRES": "gpu:a100"}, ("10k",), "must end in a count"),
        ({"SLURM_GPU_PARTITION": ""}, ("10k",), "SLURM_GPU_PARTITION is not set"),
        ({"AUTONMI_NODE_CPUS": 8}, ("1k", "stub"), "leave no workers"),
    ],
)
def test_a_site_file_missing_what_a_run_needs_is_refused(tmp_path, settings, args, why):
    r = scale(tmp_path, "dry-run", *args, env_file=site(tmp_path, **settings))
    assert r.returncode == 1 and why in r.stderr, r.stderr


def test_a_rung_on_several_nodes_needs_dragon_s_options(tmp_path):
    text = SITE.replace("AUTONMI_DRAGON_ARGS", "# AUTONMI_DRAGON_ARGS")
    r = scale(tmp_path, "dry-run", "10k", "stub", env_file=site(tmp_path, text))
    assert r.returncode == 1 and "set AUTONMI_DRAGON_ARGS" in r.stderr
    r = scale(tmp_path, "dry-run", "1k", "stub", env_file=site(tmp_path, text))
    assert r.returncode == 0, r.stderr  # one node runs dragon -s


def test_without_a_site_file_or_with_an_unknown_rung_it_says_how(tmp_path):
    r = scale(tmp_path, "dry-run", "10k", env_file=tmp_path / "missing.env")
    assert r.returncode == 1 and "copy jobs/local.env.example" in r.stderr
    r = scale(tmp_path, "dry-run", "20k")
    assert r.returncode == 2 and "RUNG: 1k, 10k, 100k or 300k" in r.stdout


def fake_slurm(tmp_path, fail_at: int = 0) -> tuple[dict, Path]:
    """sbatch printing job ids from 101 and refusing its fail_at-th call; scancel and sacct."""
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    log = tmp_path / "slurm.log"
    (bin_ / "sbatch").write_text(
        "#!/bin/sh\n"
        f'n=$(cat "{log}.n" 2>/dev/null || echo 100); n=$((n + 1)); echo $n > "{log}.n"\n'
        f'echo "sbatch $*" >> "{log}"\n'
        f'[ $((n - 100)) -eq {fail_at} ] && {{ echo "sbatch: error: refused" >&2; exit 1; }}\n'
        'echo "$n"\n'
    )
    (bin_ / "scancel").write_text(f'#!/bin/sh\necho "scancel $*" >> "{log}"\n')
    (bin_ / "sacct").write_text("#!/bin/sh\necho '       101 autonmi-dragon COMPLETED'\n")
    for tool in ("sbatch", "scancel", "sacct"):
        (bin_ / tool).chmod(0o755)
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    runs = tmp_path / "runs"
    text = SITE.replace("/opt/venv", str(tmp_path / "venv")).replace(RUNS, str(runs))
    env = {"PATH": f"{bin_}:{os.environ['PATH']}"}
    return env, site(tmp_path, text)


def test_submit_chains_the_jobs_and_records_their_ids(tmp_path):
    env, env_file = fake_slurm(tmp_path)
    r = scale(tmp_path, "submit", "10k", env_file=env_file, **env)
    assert r.returncode == 0, r.stderr
    log = (tmp_path / "slurm.log").read_text().splitlines()
    assert len(log) == 4
    assert option(log[1], "-d") == "afterok:101" and option(log[2], "-d") == "after:102"
    assert option(log[3], "-d") == "afterany:103" and log[3].endswith(
        "test=101,model=102,campaign=103"
    )
    runs = tmp_path / "runs"
    state = (runs / "scale-10k.jobs").read_text()
    assert "test=101\nmodel=102\ncampaign=103\nreport=104\n" in state
    assert f"settings={env_file}" in state
    assert all((runs / d).is_dir() for d in ("scale-10k", "scale-10k-fleet", "scale-10k-test"))
    assert "Check with:" in r.stdout and "status 10k" in r.stdout
    again = scale(tmp_path, "submit", "10k", env_file=env_file, **env)
    assert again.returncode == 1 and "already submitted" in again.stderr
    r = scale(tmp_path, "status", "10k", env_file=env_file, **env)
    assert r.returncode == 0 and "autonmi-dragon COMPLETED" in r.stdout, r.stderr
    assert "no report and no summary.json yet" in r.stdout
    r = scale(tmp_path, "cancel", "10k", env_file=env_file, **env)
    assert "scancel 101 102 103 104" in (tmp_path / "slurm.log").read_text()


def test_a_refused_job_cancels_the_ones_already_submitted(tmp_path):
    env, env_file = fake_slurm(tmp_path, fail_at=3)
    r = scale(tmp_path, "submit", "10k", env_file=env_file, **env)
    assert r.returncode == 1 and "refused the campaign job" in r.stderr
    assert "scancel 101 102" in (tmp_path / "slurm.log").read_text()
    assert not (tmp_path / "runs" / "scale-10k.jobs").exists()


def test_submit_refuses_a_run_directory_that_holds_an_earlier_try(tmp_path):
    env, env_file = fake_slurm(tmp_path)
    (tmp_path / "runs" / "scale-1k-stub").mkdir(parents=True)
    (tmp_path / "runs" / "scale-1k-stub" / "summary.json").write_text("{}")
    r = scale(tmp_path, "submit", "1k", "stub", env_file=env_file, **env)
    assert r.returncode == 1 and "holds an earlier try" in r.stderr
    assert not (tmp_path / "slurm.log").exists()


# --- the report, on run directories written here ---------------------------------------------


def write_run(run: Path, agents: int = 20, rows_of: range | None = None, summary: bool = True):
    """A run of `agents` agents in two populations of 4 workers on two nodes, as the mini-app writes
    it: every agent reports its candidate's best after two simulations of 10 s, from t=1000."""
    sys.path.insert(0, str(EXAMPLE))
    try:
        from screening import physics
        from screening.benchmark import Benchmark
    finally:
        sys.path.remove(str(EXAMPLE))
    bench = Benchmark(agents=agents)
    (run / "app").mkdir(parents=True)
    (run / "campaign.json").write_text(
        json.dumps(
            {"knobs": bench.asdict(), "config": {"populations": 2, "population": {"workers": 4}}}
        )
    )
    pops: dict[int, list] = {0: [], 1: []}
    for i in rows_of if rows_of is not None else range(agents):
        c = physics.candidate(i, bench)
        t, d = physics.optimum(c)
        pops[i % 2].append(
            {
                "agent_id": c.cid,
                "cid": c.cid,
                "sims": [[t + 5, d, f"r/{c.cid}/0"], [t, d, f"r/{c.cid}/1"]],
                "sim_seconds": [[10.0, 10.01], [10.0, 10.02]],
                "report": {"temperature": t, "doping": d, "score": physics.score(c, t, d)},
                "finished": True,
                "status": "done",
                "t_start": 1000.0 + i,
                "t_end": 1100.0 + i,
                "model_calls": 7,
                "model_wait_s": 1.0,
                "model_service_s": 3.0,
            }
        )
    for p, rows in pops.items():
        if rows:
            (run / "app" / f"agents-p{p:03d}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            pdir = run / "populations" / f"p{p:03d}"
            pdir.mkdir(parents=True)
            (pdir / "population.json").write_text(
                json.dumps({"node": f"node{p}", "agents": len(rows)})
            )
    if summary:
        checks = [
            {"name": "every_agent_finishes_and_verifies", "ok": True,
             "detail": {"verified": agents, "agents": agents}},
            {"name": "model_calls_as_scripted", "ok": True, "detail": None},
        ]  # fmt: skip
        (run / "app" / "checks.json").write_text(json.dumps(checks))
        (run / "summary.json").write_text(json.dumps({"ok": True, "checks": checks}))
    (run / "timeline").write_text(
        "job_start 900.0\ndragon_start 950.0\ndragon_end 1150.0\njob_end 1160.0\n"
    )
    (run / "jobs.txt").write_text(
        "JobID|JobName|State|Submit|Start|End|Elapsed|NNodes|NodeList\n"
        "101|autonmi-dragon|COMPLETED|2026-11-26T10:00:00|2026-11-26T10:00:30|2026-11-26T10:02:00"
        "|00:01:30|1|node0\n"
        "103|autonmi-dragon|COMPLETED|2026-11-26T10:00:00|2026-11-26T10:12:00|Unknown|00:04:20|2|"
        "node[0-1]\n"
    )


def report(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPORT), *map(str, args)], capture_output=True, text=True, timeout=60
    )


def test_the_report_verifies_every_agent_and_says_where_the_time_went(tmp_path):
    run = tmp_path / "scale-1k-stub"
    write_run(run)
    r = report("write", run)
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.loads((run / "scale-report.json").read_text())
    assert out["ok"] and out["every_agent_verifies"]
    assert (
        out["verification"]["verified"] == 20 and out["verification_by_the_run"]["verified"] == 20
    )
    t = out["time"]
    assert t["agents_s"] == 119.0 and t["startup_s"] == 50.0 and t["winddown_s"] == 31.0
    assert t["job_s"] == 260.0 and t["cleanup_s"] == 10.0
    assert t["bound_s"] == 50.0 and t["workers"] == 8  # 40 simulations of 10 s over 2 x 4 workers
    assert t["agent_s"] == 2000.0 and t["agent_s_simulations"] == pytest.approx(400.6)
    assert t["agent_s_model_admission"] == 20.0 and t["agent_s_model_serving"] == 60.0
    assert out["nodes"] == {"node0": [1, 10], "node1": [1, 10]}
    assert [j["wait_s"] for j in out["jobs"]] == [30.0, 720.0]
    text = r.stdout
    assert "scale-1k-stub: ok, every agent verifies and every check passes" in text
    assert "agents: 20 of 20 verify, from their rows; the run's own check: 20 of 20" in text
    assert "start-up 50 s" in text and "2.38 times the simulations' bound of 50 s" in text
    assert "waited 0:12:00, ran 00:04:20" in text and "140 model calls" in text
    shown = report("show", run)
    assert shown.stdout.splitlines() == out["lines"]


def test_without_a_summary_the_report_verifies_the_agents_the_rows_have(tmp_path):
    """A campaign that died: one population wrote its rows, the other did not."""
    run = tmp_path / "scale-10k-stub"
    write_run(run, rows_of=range(0, 20, 2), summary=False)
    r = report("write", run)
    assert r.returncode == 1
    out = json.loads((run / "scale-report.json").read_text())
    assert not out["ok"] and not out["summary_written"]
    assert out["verification"]["verified"] == 10
    assert out["verification"]["reasons"] == {"no agent record": 10}
    assert "the mini-app wrote no summary.json" in r.stdout
    assert "agents: 10 of 20 verify, from their rows" in r.stdout


def test_a_report_that_does_not_verify_says_why(tmp_path):
    run = tmp_path / "run"
    write_run(run)
    path = run / "app" / "agents-p001.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["report"].update(temperature=300.0, doping=0.0)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = report("write", run)
    assert r.returncode == 1
    out = json.loads((run / "scale-report.json").read_text())
    assert out["verification"]["reasons"] == {"report below tolerance": 1}
    assert "FAILED" in r.stdout or "19 of 20" in r.stdout


def test_compare_matches_the_agents_both_runs_have(tmp_path):
    small, large = tmp_path / "scale-1k-stub", tmp_path / "scale-10k-stub"
    write_run(small, agents=10)
    write_run(large, agents=20)
    r = report("compare", small, large)
    assert r.returncode == 0, r.stdout
    out = json.loads(r.stdout)
    assert out["common"] == 10 and out["identical"] == 10 and out["agents_second"] == 20
    path = large / "app" / "agents-p000.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["model_calls"] = 9
    rows[2]["report"]["t"] = 123.0  # when it reported: not part of the comparison
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = report("compare", small, large)
    out = json.loads(r.stdout)
    assert r.returncode == 1 and out["differ"] == 1 and list(out["examples"]) == ["mat-000002"]
