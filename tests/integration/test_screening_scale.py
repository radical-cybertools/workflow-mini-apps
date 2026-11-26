"""The scale ladder of the agentic screening example (``examples/agentic-screening/jobs/``).

``jobs/scale`` submits a chain of Slurm jobs; these tests run it with ``dry-run``, or with fake
``sbatch``, ``scancel``, ``squeue`` and ``sacct`` on ``PATH``, so they need no Slurm. The jobs run
with a fake ``dragon`` and fake Slurm tools, and the report (``jobs/scale_report.py``) on run
directories written here: no framework and no GPU.
"""

import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "agentic-screening"
JOBS = EXAMPLE / "jobs"
SCALE = JOBS / "scale"
REPORT = JOBS / "scale_report.py"
RUNS = "/runs"

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


def site(tmp_path, text: str = SITE, **settings) -> Path:
    path = tmp_path / "site.env"
    path.write_text(text + "".join(f"{k}={v}\n" for k, v in settings.items()))
    return path


def scale(tmp_path, *args: str, env_file: Path | None = None, **env) -> subprocess.CompletedProcess:
    """jobs/scale as a user runs it, with only the settings given: none of the caller's."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("AUTONMI_", "SLURM_"))}
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
    test, model, campaign, report = jobs["test"], jobs["model"], jobs["campaign"], jobs["report"]
    assert option(test, "-N") == "2" and option(test, "-d") == "singleton"  # Dragon on two nodes
    assert option(model, "-d") == "afterok:<test>" and option(model, "--gres") == "gpu:a100:2"
    assert option(model, "-p") == "gpu" and model.endswith(f"{RUNS}/scale-10k-fleet 2")
    assert option(campaign, "-d") == "afterok:<test>,after:<model>,singleton"
    assert option(campaign, "-N") == "2"
    assert "--exclusive" in campaign and option(campaign, "--cpus-per-task") == "96"
    assert option(campaign, "--populations") == "4" and option(campaign, "--workers") == "44"
    assert option(campaign, "--model") == "openai" and option(campaign, "--cap") == "96"
    assert option(campaign, "--endpoints-file") == f"{RUNS}/scale-10k-fleet/endpoints"
    assert f"campaign.sh {RUNS}/scale-10k {RUNS}/scale-10k-fleet <model> --agents" in campaign
    assert option(campaign, "--model-wait") == "1200"  # 600 s and 300 s per server
    assert (
        option(campaign, "--signal") == "B:USR1@240"
        and option(campaign, "-D") == f"{RUNS}/scale-10k"
    )
    assert option(report, "-d") == "afterany:<campaign>"
    assert report.endswith(
        "test=<test>,model=<model>,campaign=<campaign> --agents 10000 --populations 4 --workers 176"
    )
    for cmd in (test, campaign, report):  # run once: a requeue would start over in the same run
        assert "--no-requeue" in cmd and option(cmd, "--open-mode") == "append"
    assert "--requeue" in model.split()  # the model job may come back: the campaign waits for it
    assert all(option(cmd, "-A") == "acct" for cmd in jobs.values())
    assert all("--export=ALL" in cmd for cmd in jobs.values())


def test_with_the_stub_there_is_no_model_job(tmp_path):
    r = scale(tmp_path, "dry-run", "1k", "stub")
    assert r.returncode == 0, r.stderr
    jobs = jobs_of(r.stdout)
    assert list(jobs) == ["test", "campaign", "report"]
    campaign = jobs["campaign"]
    assert option(campaign, "-d") == "afterok:<test>,singleton"
    assert option(campaign, "--model") == "stub" and option(jobs["test"], "-d") == "singleton"
    assert (
        f"campaign.sh {RUNS}/scale-1k-stub none none " in campaign
        and "--endpoints-file" not in campaign
    )
    assert " -N1 " in jobs["test"]  # one node runs dragon -s
    assert jobs["report"].endswith(
        "test=<test>,model=,campaign=<campaign> --agents 1000 --populations 2 --workers 88"
    )


@pytest.mark.parametrize(
    "rung, agents, nodes, gpus, cap, mcp, time, timeout, model_time",
    [
        # the demo's limits, raised where the model wait, start-up and wind-down need more
        ("1k", 1000, 1, 1, 96, 4, "00:41:00", 900, "02:01:00"),
        ("10k", 10000, 2, 2, 96, 4, "01:17:00", 2700, "02:02:00"),
        ("100k", 100000, 10, 2, 320, 4, "02:00:00", 4800, "08:00:00"),
        ("300k", 300000, 30, 8, 960, 8, "02:50:00", 4800, "08:50:00"),
    ],
)
def test_each_rung_has_the_table_s_shape_on_nodes_of_96_cpus(
    tmp_path, rung, agents, nodes, gpus, cap, mcp, time, timeout, model_time
):
    """With a model fast enough that the simulations set the time (1,000 calls/s per server)."""
    r = scale(tmp_path, "dry-run", rung, env_file=site(tmp_path, AUTONMI_MODEL_CALLS_PER_S=1000))
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


def test_the_campaign_s_limit_covers_the_parts_outside_its_populations_timeout(tmp_path):
    """The model wait, start-up, wind-down, clean-up and margin: 300k with 8 servers needs 170 min,
    more than the demo's 120."""
    r = scale(tmp_path, "dry-run", "300k", env_file=site(tmp_path, AUTONMI_MODEL_CALLS_PER_S=1000))
    campaign = jobs_of(r.stdout)["campaign"]
    wait, timeout = int(option(campaign, "--model-wait")), int(option(campaign, "--timeout"))
    assert wait == 600 + 300 * 8 and timeout == 4800
    parts = wait + (120 + 10 * 30) + timeout + (120 + 300_000 // 200) + 120 + 240
    assert option(campaign, "-t") == "02:50:00" and 169 * 60 < parts <= 170 * 60
    r = scale(tmp_path, "dry-run", "300k", "stub")  # no model wait: the demo's limit is enough
    assert option(jobs_of(r.stdout)["campaign"], "-t") == "02:00:00"


def test_a_model_rung_s_limits_hold_the_model_s_time(tmp_path):
    """The model's bound: the rung's model calls over its servers' calls per second. 10k makes
    85,670 calls; at the default 10 calls/s per server, on the rung's 2 servers, that is 4,283 s,
    and the populations get half as much again."""
    r = scale(tmp_path, "dry-run", "10k")
    assert r.returncode == 0, r.stderr
    campaign, model = jobs_of(r.stdout)["campaign"], jobs_of(r.stdout)["model"]
    assert option(campaign, "--timeout") == "6424"
    assert option(campaign, "-t") == "02:19:00" and option(model, "-t") == "03:04:00"
    assert "the model's bound about 4283 s (85670 calls, 10 per second on each of 2 servers)" in (
        r.stdout
    )
    # on one GPU where the rung has two, twice as long: the 10k run on one GPU that ran out of time
    r = scale(tmp_path, "dry-run", "10k", AUTONMI_GPUS="1")
    assert option(jobs_of(r.stdout)["campaign"], "--timeout") == "12850"
    # a faster model: the simulations set the time again
    r = scale(tmp_path, "dry-run", "10k", env_file=site(tmp_path, AUTONMI_MODEL_CALLS_PER_S=40))
    assert option(jobs_of(r.stdout)["campaign"], "--timeout") == "2700"
    # the stub answers at once; AUTONMI_TIMEOUT replaces the bound
    r = scale(tmp_path, "dry-run", "10k", "stub")
    assert option(jobs_of(r.stdout)["campaign"], "--timeout") == "2700"
    assert "model's bound" not in r.stdout
    r = scale(tmp_path, "dry-run", "10k", AUTONMI_TIMEOUT="3000")
    assert option(jobs_of(r.stdout)["campaign"], "--timeout") == "3000"
    r = scale(tmp_path, "dry-run", "10k", env_file=site(tmp_path, AUTONMI_MODEL_CALLS_PER_S="2.5"))
    assert r.returncode == 1 and "AUTONMI_MODEL_CALLS_PER_S must be a whole number" in r.stderr


def test_the_benchmark_s_work_and_model_calls_per_agent_are_the_ladder_s(monkeypatch):
    """The bounds the dry run prints use 31.45 core-seconds and 8.567 model calls per agent: the
    benchmark's physics and workflow at its defaults, which the report measures from the rows."""
    monkeypatch.syspath_prepend(str(EXAMPLE))
    from screening import physics
    from screening.benchmark import Benchmark

    bench = Benchmark(agents=10_000)
    work, calls = 0.0, 0
    for i in range(bench.agents):
        c = physics.candidate(i, bench)
        t, d = physics.reference(c, bench)
        for n in range(1, bench.max_sims + 1):
            work += physics.sim_seconds(c, t, d, bench)
            _, done, nxt = physics.analysis(c, t, d, n, bench)
            if done:
                break
            t, d = nxt
        calls += bench.model_calls_per_agent(n)
    assert re.search(r"^CORE_CS_PER_AGENT=3145\b", SCALE.read_text(), re.M)
    assert work / bench.agents == pytest.approx(31.45, abs=0.01)
    assert re.search(r"^CALLS_PER_1000_AGENTS=8567\b", SCALE.read_text(), re.M)
    assert calls == 85_670  # the stub's 10k run made as many


def test_the_workers_follow_the_site_s_cpus_and_the_limits_follow_the_workers(tmp_path):
    """On nodes of 64 CPUs, shared: 28 workers per population, and limits 88 / 56 times longer."""
    env_file = site(
        tmp_path, AUTONMI_NODE_CPUS=64, AUTONMI_NODE_EXCLUSIVE=0, AUTONMI_MODEL_CALLS_PER_S=1000
    )
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
    env_file = site(
        tmp_path, AUTONMI_NODE_CPUS=192, AUTONMI_NODE_POPULATIONS=4, AUTONMI_MODEL_CALLS_PER_S=1000
    )
    r = scale(tmp_path, "dry-run", "100k", env_file=env_file)
    assert r.returncode == 0, r.stderr
    campaign = jobs_of(r.stdout)["campaign"]
    assert option(campaign, "--workers") == "46" and option(campaign, "--populations") == "40"
    assert option(campaign, "-t") == "02:00:00" and option(campaign, "--timeout") == "4800"


def test_one_run_s_settings_change_only_that_run(tmp_path):
    r = scale(
        tmp_path, "dry-run", "10k", "stub",
        AUTONMI_NODES="1", AUTONMI_TAG="1n", AUTONMI_TIME="00:30:00", AUTONMI_TIMEOUT="600",
        AUTONMI_ARGS="--span-ratio 1 --seed 3",
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    campaign = jobs_of(r.stdout)["campaign"]
    assert option(campaign, "-N") == "1" and option(campaign, "--populations") == "2"
    assert option(campaign, "-t") == "00:30:00" and option(campaign, "--timeout") == "600"
    assert campaign.endswith("--model stub --span-ratio 1 --seed 3")
    assert option(campaign, "-o") == f"{RUNS}/scale-10k-stub-1n/campaign-%j.out"
    r = scale(tmp_path, "dry-run", "10k", AUTONMI_GPUS="1")
    assert option(jobs_of(r.stdout)["model"], "--gres") == "gpu:a100:1"
    # names without the prefix, such as a stray TIMEOUT in the shell, change nothing
    r = scale(tmp_path, "dry-run", "10k", "stub", TIMEOUT="30", NODES="1", ARGS="--seed 3")
    campaign = jobs_of(r.stdout)["campaign"]
    assert option(campaign, "--timeout") == "2700" and option(campaign, "-N") == "2"
    assert "--seed" not in campaign


@pytest.mark.parametrize("args", ["--agents 100", "--populations=3", "--workers 2", "--run-dir /x"])
def test_one_run_s_arguments_may_not_change_the_rung_s_shape(tmp_path, args):
    r = scale(tmp_path, "dry-run", "1k", "stub", AUTONMI_ARGS=args)
    assert r.returncode == 1 and "may not set" in r.stderr
    r = scale(
        tmp_path,
        "dry-run",
        "1k",
        "stub",
        env_file=site(tmp_path, AUTONMI_CAMPAIGN_ARGS=f'"{args}"'),
    )
    assert r.returncode == 1 and "may not set" in r.stderr


@pytest.mark.parametrize("name, value", [("AUTONMI_NODES", "0"), ("AUTONMI_GPUS", "two")])
def test_one_run_s_counts_must_be_whole_numbers(tmp_path, name, value):
    r = scale(tmp_path, "dry-run", "1k", **{name: value})
    assert r.returncode == 1 and f"{name} must be a whole number" in r.stderr


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
        AUTONMI_DRAGON_SINGLETON=0,
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
    assert " -d " not in jobs["test"]
    assert option(jobs["campaign"], "-d") == "afterok:<test>,after:<model>"


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


def fake_slurm(tmp_path, fail_at: int = 0, hang_at: int = 0, runs: Path | None = None):
    """sbatch printing job ids from 101, refusing its fail_at-th call and hanging on its hang_at-th;
    scancel logging its calls; squeue knowing no job; sacct. The environment and the site file."""
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    log = tmp_path / "slurm.log"
    (bin_ / "sbatch").write_text(
        "#!/bin/sh\n"
        f'n=$(cat "{log}.n" 2>/dev/null || echo 100); n=$((n + 1)); echo $n > "{log}.n"\n'
        f'echo "sbatch $*" >> "{log}"\n'
        f'[ $((n - 100)) -eq {fail_at} ] && {{ echo "sbatch: error: refused" >&2; exit 1; }}\n'
        f'[ $((n - 100)) -eq {hang_at} ] && {{ touch "{log}.hang"; exec sleep 30; }}\n'
        'echo "$n"\n'
    )
    (bin_ / "scancel").write_text(f'#!/bin/sh\necho "scancel $*" >> "{log}"\n')
    (bin_ / "sacct").write_text("#!/bin/sh\necho '       101 autonmi-dragon COMPLETED'\n")
    (bin_ / "squeue").write_text("#!/bin/sh\nexit 0\n")
    for tool in ("sbatch", "scancel", "sacct", "squeue"):
        (bin_ / tool).chmod(0o755)
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True, exist_ok=True)
    if not (venv / "python").exists():
        (venv / "python").symlink_to(sys.executable)
    runs = runs or tmp_path / "runs"
    text = SITE.replace("/opt/venv", str(tmp_path / "venv")).replace(RUNS, f'"{runs}"')
    env = {"PATH": f"{bin_}:{os.environ['PATH']}"}
    return env, site(tmp_path, text)


def state_of(path: Path) -> dict:
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def test_submit_chains_the_jobs_and_records_their_ids_as_it_goes(tmp_path):
    env, env_file = fake_slurm(tmp_path)
    r = scale(tmp_path, "submit", "10k", env_file=env_file, **env)
    assert r.returncode == 0, r.stderr
    log = (tmp_path / "slurm.log").read_text().splitlines()
    assert len(log) == 4
    assert option(log[1], "-d") == "afterok:101"
    assert option(log[2], "-d") == "afterok:101,after:102,singleton"
    assert " 102 --agents 10000 " in log[2]  # the campaign checks the model job itself
    assert option(log[3], "-d") == "afterany:103"
    assert "test=101,model=102,campaign=103 --agents 10000" in log[3]
    runs = tmp_path / "runs"
    state = state_of(runs / "scale-10k.jobs")
    assert {k: state[k] for k in ("test", "model", "campaign", "report")} == {
        "test": "101", "model": "102", "campaign": "103", "report": "104"
    }  # fmt: skip
    assert state["settings"] == str(env_file) and "cancelled" not in state
    assert not (runs / "scale-10k.jobs.lock").exists()
    assert all((runs / d).is_dir() for d in ("scale-10k", "scale-10k-fleet", "scale-10k-test"))
    assert f"Check with: AUTONMI_LOCAL_ENV={env_file} {SCALE} status 10k\n" in r.stdout
    again = scale(tmp_path, "submit", "10k", env_file=env_file, **env)
    assert again.returncode == 1 and "already submitted" in again.stderr
    r = scale(tmp_path, "status", "10k", env_file=env_file, **env)
    assert r.returncode == 0 and "autonmi-dragon COMPLETED" in r.stdout, r.stderr
    assert "no report and no summary.json yet" in r.stdout
    r = scale(tmp_path, "cancel", "10k", env_file=env_file, **env)
    assert r.returncode == 0, r.stderr
    log = (tmp_path / "slurm.log").read_text()
    assert "scancel 101 102 103\n" in log  # the report job records the run, then ends the model job
    assert "-s USR1" not in log  # nothing was running


def test_a_refused_job_cancels_the_ones_already_submitted(tmp_path):
    env, env_file = fake_slurm(tmp_path, fail_at=3)
    r = scale(tmp_path, "submit", "10k", env_file=env_file, **env)
    assert r.returncode == 1 and "refused the campaign job" in r.stderr
    assert "scancel 101 102" in (tmp_path / "slurm.log").read_text()
    assert not (tmp_path / "runs" / "scale-10k.jobs").exists()  # a submit may start over
    assert not (tmp_path / "runs" / "scale-10k.jobs.lock").exists()


def test_sbatch_output_that_is_not_a_job_id_counts_as_a_refusal(tmp_path):
    env, env_file = fake_slurm(tmp_path)
    (tmp_path / "bin" / "sbatch").write_text("#!/bin/sh\necho 'Submitted batch job 5'\n")
    r = scale(tmp_path, "submit", "1k", "stub", env_file=env_file, **env)
    assert r.returncode == 1 and "refused the test job" in r.stderr


def test_an_interrupted_submit_cancels_what_it_submitted_and_says_so(tmp_path):
    """Ctrl-C, or an ssh connection that drops, while the chain is half submitted."""
    env, env_file = fake_slurm(tmp_path, hang_at=2)
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("AUTONMI_", "SLURM_"))}
    clean.update(env, AUTONMI_LOCAL_ENV=str(env_file))
    job = subprocess.Popen(
        ["bash", str(SCALE), "submit", "10k"], env=clean, start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        for _ in range(200):
            if (tmp_path / "slurm.log.hang").exists():
                break
            time.sleep(0.05)
        os.killpg(job.pid, signal.SIGINT)  # as a terminal does: the whole foreground group
        _, err = job.communicate(timeout=30)
    finally:
        job.kill()
    assert job.returncode != 0 and "interrupted while submitting" in err
    assert "scancel 101" in (tmp_path / "slurm.log").read_text()
    state = state_of(tmp_path / "runs" / "scale-10k.jobs")
    assert state["test"] == "101" and "interrupted" in state["cancelled"]
    assert not (tmp_path / "runs" / "scale-10k.jobs.lock").exists()
    r = scale(tmp_path, "status", "10k", env_file=env_file, **env)
    assert "cancelled: interrupted while submitting: 101" in r.stdout


def test_two_submits_of_the_same_rung_do_not_both_run(tmp_path):
    env, env_file = fake_slurm(tmp_path)
    lock = tmp_path / "runs" / "scale-1k-stub.jobs.lock"
    lock.mkdir(parents=True)  # another submit holds it
    r = scale(tmp_path, "submit", "1k", "stub", env_file=env_file, **env)
    assert r.returncode == 1 and "another submit of 1k-stub runs" in r.stderr
    assert lock.is_dir() and not (tmp_path / "slurm.log").exists()  # its lock is not ours to free
    lock.rmdir()
    r = scale(tmp_path, "submit", "1k", "stub", env_file=env_file, **env)
    assert r.returncode == 0, r.stderr
    assert not lock.exists()


def test_paths_with_spaces_submit_report_status_and_cancel(tmp_path):
    runs = tmp_path / "my runs"
    env, env_file = fake_slurm(tmp_path, runs=runs)
    moved = tmp_path / "my site" / "site 1.env"
    moved.parent.mkdir()
    env_file.rename(moved)
    r = scale(tmp_path, "submit", "1k", "stub", env_file=moved, **env)
    assert r.returncode == 0, r.stderr
    assert "AUTONMI_LOCAL_ENV=" in r.stdout and "my\\ site/site\\ 1.env" in r.stdout  # quoted
    assert state_of(runs / "scale-1k-stub.jobs")["settings"] == str(moved)
    r = scale(tmp_path, "status", "1k", "stub", env_file=moved, **env)
    assert r.returncode == 0 and "1k-stub submitted" in r.stdout, r.stderr
    r = scale(tmp_path, "cancel", "1k", "stub", env_file=moved, **env)
    assert r.returncode == 0 and "scancel 101 102" in (tmp_path / "slurm.log").read_text()


def test_the_state_file_is_read_never_run(tmp_path):
    env, env_file = fake_slurm(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "scale-1k-stub.jobs").write_text(
        f"test=101\ncampaign=102\nsettings=$(touch {tmp_path}/ran)\ncommit=`touch {tmp_path}/ran`\n"
    )
    r = scale(tmp_path, "status", "1k", "stub", env_file=env_file, **env)
    assert r.returncode == 0, r.stderr
    r = scale(tmp_path, "cancel", "1k", "stub", env_file=env_file, **env)
    assert r.returncode == 0 and not (tmp_path / "ran").exists()
    assert "scancel 101 102" in (tmp_path / "slurm.log").read_text()


def test_cancel_lets_a_running_dragon_job_end_its_run_and_clean_up_first(tmp_path):
    env, env_file = fake_slurm(tmp_path)
    assert scale(tmp_path, "submit", "1k", "stub", env_file=env_file, **env).returncode == 0
    ended = tmp_path / "ended"
    log = tmp_path / "slurm.log"
    (tmp_path / "bin" / "squeue").write_text(
        "#!/bin/sh\n"  # the campaign (102) runs until its batch shell gets USR1
        f'[ -e "{ended}" ] && exit 0\n'
        'case "$*" in *"-j 102 -o %T"*) echo RUNNING ;; *"-j 102"*) echo " 102" ;; esac\n'
    )
    (tmp_path / "bin" / "scancel").write_text(
        f'#!/bin/sh\necho "scancel $*" >> "{log}"\n[ "$1" = -b ] && touch "{ended}"\nexit 0\n'
    )
    r = scale(tmp_path, "cancel", "1k", "stub", env_file=env_file, AUTONMI_CANCEL_WAIT="20", **env)
    assert r.returncode == 0, r.stderr
    calls = [c for c in log.read_text().splitlines() if c.startswith("scancel")]
    assert calls == ["scancel -b -s USR1 102", "scancel 101 102"]
    assert "end their run and clean up" in r.stdout


def test_submit_refuses_a_run_directory_that_holds_an_earlier_try(tmp_path):
    env, env_file = fake_slurm(tmp_path)
    (tmp_path / "runs" / "scale-1k-stub").mkdir(parents=True)
    (tmp_path / "runs" / "scale-1k-stub" / "summary.json").write_text("{}")
    r = scale(tmp_path, "submit", "1k", "stub", env_file=env_file, **env)
    assert r.returncode == 1 and "holds an earlier try" in r.stderr
    assert not (tmp_path / "slurm.log").exists()


# --- the report, on run directories written here ---------------------------------------------


PIN = "cdda559fcd78dbf42671305c3fdb0b6fa97d451b"


def write_run(
    run: Path,
    agents: int = 20,
    rows_of: range | None = None,
    summary: bool = True,
    autonmi: str = PIN,
):
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
    pin = {"commit": autonmi, "dirty": False, "pin": PIN, "matches_pin": autonmi == PIN}
    (run / "autonmi.json").write_text(json.dumps(pin))
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
    r = report("write", run, "--jobs", "test=101,model=,campaign=103")
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
    assert "test COMPLETED, waited 0:00:30, ran 00:01:30; campaign COMPLETED, waited 0:12" in text
    assert "140 model calls" in text
    assert "autonmi: cdda559fcd78, the pinned commit" in text
    shown = report("show", run)
    assert shown.stdout.splitlines() == out["lines"]


def test_without_a_summary_the_report_verifies_the_agents_the_rows_have(tmp_path):
    """A campaign that died: one population wrote its rows, the other did not."""
    run = tmp_path / "scale-10k-stub"
    write_run(run, rows_of=range(0, 20, 2), summary=False, autonmi="6c1ff80")
    r = report("write", run)
    assert r.returncode == 1
    out = json.loads((run / "scale-report.json").read_text())
    assert not out["ok"] and not out["summary_written"]
    assert out["verification"]["verified"] == 10
    assert out["verification"]["reasons"] == {"no agent record": 10}
    assert "the mini-app wrote no summary.json" in r.stdout
    assert "agents: 10 of 20 verify, from their rows" in r.stdout
    assert out["populations_without_rows"] == ["p001"]
    assert "populations that wrote no rows: p001" in r.stdout
    assert "autonmi: 6c1ff80, NOT the pinned cdda559fcd78" in r.stdout


def test_a_run_of_another_size_than_the_rung_fails(tmp_path):
    """AUTONMI_ARGS cannot change --agents any more; the report checks it all the same."""
    run = tmp_path / "scale-1k-stub"
    write_run(run)  # 20 agents, every one verified
    r = report("write", run, "--agents", "1000", "--populations", "2")
    assert r.returncode == 1
    out = json.loads((run / "scale-report.json").read_text())
    assert not out["ok"] and out["agents_started"] == 20 and out["agents_expected"] == 1000
    assert "the run was started with 20 agents, but the rung has 1,000" in r.stdout
    assert "agents: 20 of 1,000 verify" in r.stdout


def test_a_campaign_that_died_before_its_first_file_reports_0_of_n(tmp_path):
    run = tmp_path / "scale-1k-stub"
    write_run(run, summary=False)
    (run / "campaign.json").unlink()
    for path in (run / "app").glob("agents-p*.jsonl"):
        path.unlink()
    r = report("write", run, "--agents", "20", "--populations", "2")
    assert r.returncode == 1
    assert "agents: 0 of 20 verify" in r.stdout
    assert "populations that wrote no rows: p000, p001" in r.stdout
    r = report("write", run)  # nothing says how many: it says so, and writes the report
    assert r.returncode == 1 and "nothing to verify against" in r.stdout
    assert (run / "scale-report.json").exists()


def otel_export(pop: str, states: dict, sims: int, calls: int, seconds: dict, t: float) -> dict:
    """One telemetry export of a population, as Flowgentic writes it (otel/p*/metrics.jsonl): the
    counts so far, from t=1000 s to `t`; a request waited 500 s for admission and took 7 s."""
    fleet = {"flowgentic.fleet.name": pop, "flowgentic.fleet.replica": 0}
    t0, t1 = 1000 * 10**9, int(t * 10**9)

    def metric(name, points):
        return {"name": name, "data": {"data_points": points, "aggregation_temporality": 2}}

    def hist(attrs, count, mean):
        return {"attributes": attrs, "start_time_unix_nano": t0, "time_unix_nano": t1,
                "count": count, "sum": mean * count}  # fmt: skip

    def tool(name, outcome, count):
        attrs = {"gen_ai.tool.name": name, "flowgentic.tool.outcome": outcome}
        return hist(attrs, count, 1.0)

    in_request = states.get("model_request", 0)
    metrics = [
        metric("flowgentic.fleet.admission.wait", [hist(fleet, calls + in_request, 500.0)]),
        metric("flowgentic.fleet.call.duration", [hist(fleet, calls, 507.0)]),
        metric("flowgentic.fleet.request.duration", [hist(fleet, calls, 7.0)]),
        metric(
            "flowgentic.tool.duration",
            [
                tool("lookup_reference", "ok", 10),
                tool("simulate", "ok", sims),
                tool("simulate", "error", 1),
                tool("analyze", "ok", sims - 2),
            ],
        ),  # fmt: skip
        metric(
            "flowgentic.agents",
            [{"attributes": {"flowgentic.agent.state": s}, "value": n} for s, n in states.items()],
        ),
        metric(
            "flowgentic.agent.time",
            [{"attributes": {"flowgentic.agent.state": s}, "value": v} for s, v in seconds.items()],
        ),
    ]
    resource = {"attributes": {"autonmi.population": pop}}
    scope = {"scope": {"name": "flowgentic"}, "metrics": metrics}
    return {"resource_metrics": [{"resource": resource, "scope_metrics": [scope]}]}


def write_telemetry(run: Path) -> None:
    """Two populations of 10 agents, cut off while most waited for the model: an early export and
    the last; p001 was killed while it wrote a third."""
    seconds = {"pending": 25.0, "working": 25.0, "model_queued": 4000.0, "model_request": 200.0,
               "tool_wait": 750.0, "done": 150.0}  # fmt: skip
    last = {
        "p000": ({"done": 3, "model_queued": 6, "model_request": 1, "tool_wait": 0}, 25, 70),
        "p001": ({"done": 2, "model_queued": 7, "model_request": 1, "tool_wait": 0}, 24, 68),
    }
    for pop, (states, sims, calls) in last.items():
        early = otel_export(pop, {"model_queued": 9, "model_request": 1}, 5, 20, seconds, 1050.0)
        lines = [
            json.dumps(early),
            json.dumps(otel_export(pop, states, sims, calls, seconds, 1100)),
        ]
        if pop == "p001":
            lines.append(json.dumps(early)[:40])
        (run / "otel" / pop).mkdir(parents=True)
        (run / "otel" / pop / "metrics.jsonl").write_text("\n".join(lines) + "\n")


def test_when_populations_time_out_the_report_counts_from_telemetry(tmp_path):
    """Populations write their rows when they end: cut off, they write none, and the counts come
    from the framework's telemetry, which says where the agents were."""
    run = tmp_path / "scale-10k"
    write_run(run, rows_of=range(0), summary=False)
    write_telemetry(run)
    r = report("write", run, "--agents", "20", "--populations", "2")
    assert r.returncode == 1
    out = json.loads((run / "scale-report.json").read_text())
    assert out["counts_from"] == "telemetry"
    assert out["simulations"] == 49 and out["model_calls"] == 138  # from the last whole exports
    tel = out["telemetry"]
    assert tel["agents"] == {"done": 5, "model_queued": 13, "model_request": 2, "tool_wait": 0}
    assert tel["admission_mean_s"] == 500.0 and tel["request_mean_s"] == 7.0
    assert tel["covers_s"] == 100.0 and sorted(tel["populations"]) == ["p000", "p001"]
    text = r.stdout
    assert "agents: 0 of 20 verify, from their rows" in text
    assert "populations that wrote no rows: p000, p001" in text
    assert (
        "agents at the end, from telemetry: 5 of 20 done; model_queued 13, model_request 2" in text
    )
    assert (
        "    49 simulations, 138 model calls (from telemetry: populations without rows), "
        "1.4 calls/s over the 100 s it covers"
    ) in text
    assert (
        "    model calls: a request waited 500 s on average for admission, then took 7.0 s" in text
    )
    assert (
        "    agents' own time, from telemetry: model_queued 80.0 %, tool_wait 15.0 %, "
        "model_request 4.0 %, pending 0.5 %, working 0.5 %"
    ) in text
    assert "0 simulations" not in text


def test_when_every_population_wrote_its_rows_the_counts_are_the_rows(tmp_path):
    run = tmp_path / "scale-1k"
    write_run(run)
    write_telemetry(run)
    r = report("write", run)
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.loads((run / "scale-report.json").read_text())
    assert out["counts_from"] == "rows" and out["model_calls"] == 140 and out["simulations"] == 40
    assert "40 simulations, 140 model calls, " in r.stdout and "from telemetry" not in r.stdout
    assert "model calls: a request waited 500 s on average for admission, then took 7.0 s" in (
        r.stdout
    )


def test_duplicate_rows_and_rows_of_other_agents_fail_the_run(tmp_path):
    run = tmp_path / "run"
    write_run(run)
    path = run / "app" / "agents-p000.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    stranger = dict(rows[0], cid="mat-000999", agent_id="mat-000999")
    path.write_text("".join(json.dumps(r) + "\n" for r in [*rows, rows[0], rows[0], stranger]))
    r = report("write", run)
    assert r.returncode == 1
    v = json.loads((run / "scale-report.json").read_text())["verification"]
    assert v["verified"] == 20 and v["duplicate_rows"] == 2 and v["rows_outside_the_run"] == 1
    assert "duplicate rows: 2, for example ['mat-000000']" in r.stdout
    assert "rows of agents outside the run: 1, for example ['mat-000999']" in r.stdout


def test_rows_the_report_cannot_use_are_counted_not_fatal(tmp_path):
    run = tmp_path / "run"
    write_run(run)
    path = run / "app" / "agents-p001.jsonl"
    rows = path.read_text().splitlines()
    broken = [json.loads(rows[0]), json.loads(rows[1])]
    del broken[0]["sims"]
    broken[1]["report"] = {"temperature": "hot"}
    path.write_text(
        "\n".join([json.dumps(broken[0]), json.dumps(broken[1]), *rows[2:], "{not json"])
    )
    r = report("write", run)
    assert r.returncode == 1, r.stderr  # two agents without a usable row
    out = json.loads((run / "scale-report.json").read_text())
    assert out["unreadable_rows"] == 3 and out["verification"]["verified"] == 18


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


# --- the model and report jobs, with fake vllm, curl and sacct --------------------------------


def job_env(tmp_path, **settings) -> dict:
    """A job's environment as jobs/scale gives it: the site's file, this tree, a fake venv; and
    fake Slurm tools (no job of ours anywhere, a limit far ahead, srun and timeout that run nothing
    and their command)."""
    tool(tmp_path, "squeue", 'case "$*" in *%e*) echo 2099-01-01T00:00:00 ;; esac\n')
    tool(tmp_path, "srun", "exit 0\n")
    tool(tmp_path, "timeout", '[ "$1" = -k ] && shift 2\nshift\nexec "$@"\n')
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True, exist_ok=True)
    (venv / "activate").write_text(f'export PATH="{venv}:$PATH"\n')
    if not (venv / "python").exists():
        (venv / "python").symlink_to(sys.executable)
    text = SITE.replace("/opt/venv", str(tmp_path / "venv")).replace(RUNS, str(tmp_path / "runs"))
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("AUTONMI_", "SLURM_", "CUDA_", "PYTHON"))
    }
    env.update(
        AUTONMI_REPO=str(REPO),
        AUTONMI_LOCAL_ENV=str(site(tmp_path, text, **settings)),
        SLURM_JOB_ID="7",
        PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
    )
    return env


def tool(tmp_path, name: str, body: str) -> None:
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    (bin_ / name).write_text("#!/bin/sh\n" + body)
    (bin_ / name).chmod(0o755)


def test_the_model_job_starts_a_server_per_gpu_lists_them_and_ends_on_its_stop_file(tmp_path):
    log = tmp_path / "vllm.log"
    tool(tmp_path, "vllm", f'echo "$CUDA_VISIBLE_DEVICES $*" >> "{log}"\nexec sleep 300\n')
    tool(tmp_path, "curl", "exit 0\n")
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    env = job_env(tmp_path, AUTONMI_MODEL="/models/qwen", AUTONMI_MODEL_NAME="Q")
    env["CUDA_VISIBLE_DEVICES"] = "4,5,6"
    job = subprocess.Popen(
        ["bash", str(JOBS / "model.sh"), str(fleet), "2"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(300):  # the servers answer at once here, maybe before they log
            started = log.exists() and len(log.read_text().splitlines()) == 2
            if job.poll() is not None or ((fleet / "endpoints").exists() and started):
                break
            time.sleep(0.1)
        endpoints = (fleet / "endpoints").read_text().split()
        assert len(endpoints) == 2 and all(e.startswith("http://") for e in endpoints)
        servers = log.read_text().splitlines()
        # the first 2 of the job's GPUs, one each (a server may log after the next one starts)
        assert sorted(s.split()[0] for s in servers) == ["4", "5"]
        assert all("serve /models/qwen --served-model-name Q" in s for s in servers)
        assert (fleet / "api-key").stat().st_mode & 0o077 == 0
        (fleet / "stop").touch()
        out, _ = job.communicate(timeout=60)
    finally:
        job.kill()
    assert job.returncode == 0, out
    assert "servers: 2, GPUs per server: 1 (4 5)" in out and "stopping at" in out


def test_the_report_job_keeps_the_accounting_writes_the_report_and_ends_the_model_job(tmp_path):
    run = tmp_path / "scale-1k"
    write_run(run)
    (run / "jobs.txt").unlink()
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    calls = tmp_path / "calls.log"
    env = job_env(tmp_path)
    tool(  # the accounting lags: the campaign still runs on its first call
        tmp_path,
        "sacct",
        f'echo x >> "{calls}.sacct"\n'
        'echo "JobID|JobName|State|Submit|Start|End|Elapsed|NNodes|NodeList"\n'
        f'[ $(wc -l < "{calls}.sacct") -lt 2 ] && state=RUNNING || state=COMPLETED\n'
        'echo "103|autonmi-dragon|$state|2026-11-26T10:00:00|2026-11-26T10:01:00"'
        '"|x|00:04:20|2|n"\n',
    )
    tool(tmp_path, "squeue", 'echo " 102 autonmi-vllm"\n')  # the model job never ends by itself
    tool(tmp_path, "scancel", f'echo "scancel $*" >> "{calls}"\n')
    tool(tmp_path, "sleep", "exit 0\n")
    r = subprocess.run(
        [
            "bash",
            str(JOBS / "report.sh"),
            str(run),
            str(fleet),
            "test=101,model=102,campaign=103",
            "--agents",
            "20",
            "--populations",
            "2",
        ],  # fmt: skip
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert (fleet / "stop").exists() and "scancel 102" in calls.read_text()
    out = json.loads((run / "scale-report.json").read_text())
    assert out["ok"] and out["jobs"][0]["role"] == "campaign" and out["jobs"][0]["wait_s"] == 60.0
    assert "campaign COMPLETED, waited 0:01:00, ran 00:04:20" in r.stdout
    assert len((tmp_path / "calls.log.sacct").read_text().split()) == 2
    assert out["agents_expected"] == 20


def test_the_model_job_refuses_fewer_gpus_than_asked(tmp_path):
    tool(tmp_path, "vllm", "exit 9\n")
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    env = job_env(tmp_path, AUTONMI_MODEL="/models/qwen")
    env["CUDA_VISIBLE_DEVICES"] = "4,5,6"
    r = subprocess.run(
        ["bash", str(JOBS / "model.sh"), str(fleet), "4"], env=env, capture_output=True, text=True,
        timeout=60,
    )  # fmt: skip
    assert r.returncode == 1 and "asked for 4 GPUs, but the job has 3" in r.stdout
    assert not (fleet / "endpoints").exists()


def test_the_model_job_names_its_host_as_the_site_says_and_moves_off_a_taken_port(tmp_path):
    alive, starts = tmp_path / "alive", tmp_path / "starts"
    tool(  # the first server finds its port taken; the next one runs
        tmp_path,
        "vllm",
        f'echo x >> "{starts}"\n'
        f'[ $(wc -l < "{starts}") -eq 1 ] && {{ echo "OSError: [Errno 98] Address already in use"; '
        "exit 1; }\n"
        f'touch "{alive}"\nexec sleep 300\n',
    )
    tool(tmp_path, "curl", f'[ -e "{alive}" ]\n')
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    env = job_env(tmp_path, AUTONMI_MODEL="/models/qwen", AUTONMI_ENDPOINT_HOST="%h-ib")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    job = subprocess.Popen(
        ["bash", str(JOBS / "model.sh"), str(fleet), "1"], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )  # fmt: skip
    try:
        for _ in range(300):
            if job.poll() is not None or (fleet / "endpoints").exists():
                break
            time.sleep(0.1)
        (fleet / "stop").touch()
        out, _ = job.communicate(timeout=60)
    finally:
        job.kill()
    short = subprocess.run(["hostname", "-s"], capture_output=True, text=True).stdout.strip()
    assert job.returncode == 0, out
    assert "was taken; trying another" in out
    (endpoint,) = (fleet / "endpoints").read_text().split()
    assert endpoint.startswith(f"http://{short}-ib:")


# --- the Dragon jobs, with a fake dragon ------------------------------------------------------


FAKE_DRAGON = """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
Path(os.environ["FAKE_DRAGON_LOG"]).write_text(" ".join(args))
run = Path(args[args.index("--run-dir") + 1])
if os.environ.get("FAKE_DRAGON") == "ok":
    (run / "summary.json").write_text(json.dumps({"ok": True}))
sys.exit(0)  # dragon can exit 0 when its head process failed: the summary decides
"""


def dragon_env(tmp_path, mode: str = "ok", **settings) -> dict:
    """A Dragon job's environment, whose `python -m dragon` is a fake that writes the run's summary
    (mode ok) or exits 0 without one, as dragon does when the mini-app fails."""
    overlay = tmp_path / "overlay" / "dragon"
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / "__init__.py").write_text("")
    (overlay / "__main__.py").write_text(FAKE_DRAGON)
    env = job_env(tmp_path, AUTONMI_OVERLAY=str(tmp_path / "overlay"), **settings)
    env.update(FAKE_DRAGON=mode, FAKE_DRAGON_LOG=str(tmp_path / "dragon.log"), USER="me")
    return env


def run_job(script: str, env: dict, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(JOBS / script), *map(str, args)], env=env, capture_output=True, text=True,
        timeout=60,
    )  # fmt: skip


def test_the_test_job_passes_only_when_the_run_s_summary_says_ok(tmp_path):
    run = tmp_path / "test"
    run.mkdir()
    r = run_job("test.sh", dragon_env(tmp_path, mode="failed"), run)
    assert r.returncode == 1 and "the test failed" in r.stdout, r.stdout + r.stderr
    assert "exited 0" in r.stdout  # what dragon said, which is not enough
    r = run_job("test.sh", dragon_env(tmp_path, mode="ok"), run)
    assert r.returncode == 0 and "the test passed" in r.stdout, r.stdout + r.stderr
    assert (tmp_path / "dragon.log").read_text().startswith("-s ")


def test_on_two_nodes_the_test_runs_dragon_with_the_campaign_s_options(tmp_path):
    run = tmp_path / "test"
    run.mkdir()
    env = dragon_env(tmp_path)
    env["SLURM_JOB_NUM_NODES"] = "2"
    r = run_job("test.sh", env, run)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "dragon.log").read_text().startswith("-t tcp --network-prefix ^ib\\w*$ ")


def test_a_dragon_job_stops_when_another_of_ours_runs_on_its_node(tmp_path):
    run = tmp_path / "test"
    run.mkdir()
    env = dragon_env(tmp_path)
    tool(tmp_path, "squeue", 'case "$*" in *RUNNING*) echo "9 autonmi-dragon" ;; esac\n')
    r = run_job("test.sh", env, run)
    assert r.returncode == 3 and "another Dragon job of ours" in r.stderr
    assert not (tmp_path / "dragon.log").exists()


class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        return None


def test_the_campaign_starts_once_the_model_job_runs_and_its_servers_answer(tmp_path):
    run, fleet = tmp_path / "run", tmp_path / "fleet"
    run.mkdir()
    fleet.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        (fleet / "endpoints").write_text(f"http://127.0.0.1:{server.server_address[1]}\n")
        env = dragon_env(tmp_path, AUTONMI_MODEL_POLL="0.1")
        tool(
            tmp_path,
            "squeue",
            'case "$*" in *%e*) echo 2099-01-01T00:00:00 ;; *%T*) echo RUNNING ;; esac\n',
        )
        r = run_job("campaign.sh", env, run, fleet, 102, "--agents", 10)
    finally:
        server.shutdown()
        server.server_close()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "the model job's servers answer" in r.stdout and (fleet / "stop").exists()
    marks = [line.split()[0] for line in (run / "timeline").read_text().splitlines()]
    assert marks == ["job_start", "dragon_start", "dragon_end", "job_end"]


@pytest.mark.parametrize(
    "state, endpoints, said",
    [
        ("PENDING", False, "the model job 102 is PENDING: not starting the campaign"),
        ("", False, "the model job 102 is no longer in the queue"),
        ("RUNNING", False, "did not all answer within 1 s"),
    ],
)
def test_the_campaign_fails_at_once_when_the_model_job_is_not_serving(
    tmp_path, state, endpoints, said
):
    run, fleet = tmp_path / "run", tmp_path / "fleet"
    run.mkdir()
    fleet.mkdir()
    with socket.socket() as s:  # a port nothing answers on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    (fleet / "endpoints").write_text(f"http://127.0.0.1:{port}\n")
    env = dragon_env(tmp_path, AUTONMI_MODEL_POLL="0.2")
    env["AUTONMI_MODEL_WAIT"] = "1"
    tool(
        tmp_path,
        "squeue",
        f'case "$*" in *%e*) echo 2099-01-01T00:00:00 ;; *%T*) echo "{state}" ;; esac\n',
    )
    t0 = time.time()
    r = run_job("campaign.sh", env, run, fleet, 102, "--agents", 10)
    assert r.returncode == 4 and said in r.stdout, r.stdout + r.stderr
    assert time.time() - t0 < 30 and not (tmp_path / "dragon.log").exists()
