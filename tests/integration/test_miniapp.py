"""MiniApp behaviour; each scenario runs in its own process.

Every test runs on the Concurrent backend.  With ``WFMINI_TEST_DRAGON=1`` the same tests
also run on the real Dragon backend, under the ``dragon`` launcher (marker ``dragon``).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCENARIOS = Path(__file__).resolve().parent / "scenarios.py"
DRAGON = shutil.which("dragon", path=str(Path(sys.executable).parent)) or shutil.which("dragon")


@pytest.fixture(
    params=[
        "concurrent",
        pytest.param(
            "dragon",
            marks=[
                pytest.mark.dragon,
                pytest.mark.skipif(
                    os.environ.get("WFMINI_TEST_DRAGON") != "1" or DRAGON is None,
                    reason="set WFMINI_TEST_DRAGON=1 on a host with the dragon launcher",
                ),
            ],
        ),
    ]
)
def backend(request):
    return request.param


def run(scenario, tmp_path, backend):
    env = dict(os.environ, WFMINI_SCENARIO_BACKEND=backend)
    if backend == "dragon":
        cmd = [DRAGON, "-s", str(SCENARIOS), scenario, str(tmp_path)]
        env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    else:
        cmd = [sys.executable, str(SCENARIOS), scenario, str(tmp_path)]
    proc = subprocess.Popen(  # cwd: Dragon writes runtime files there
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=tmp_path
    )
    try:
        out, err = proc.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        import psutil  # installed with rhapsody-py[telemetry]

        for child in psutil.Process(proc.pid).children(recursive=True):  # Dragon workers too
            child.kill()
        proc.kill()
        out, err = proc.communicate()
        pytest.fail(
            f"{scenario} on {backend} did not exit within 300 s\n{out[-2000:]}\n{err[-2000:]}"
        )
    proc = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc, json.loads(lines[-1][len("RESULT ") :])


def checkpoint_events(tmp_path):
    (path,) = (tmp_path / "telemetry").glob("*.telemetry.jsonl")
    return [
        json.loads(ln)["event_type"]
        for ln in path.read_text().splitlines()
        if '"section": "event"' in ln
    ]


def test_ok_run_is_complete_and_blocks_are_not_tasks(tmp_path, backend):
    proc, res = run("ok", tmp_path, backend)
    assert proc.returncode == 0
    report = res["report"]
    assert report["telemetry_complete"], report["reasons"]
    assert report["tasks"]["created"] == 3 == report["tasks"]["completed"]
    events = checkpoint_events(tmp_path)
    assert events.count("wfminiapi.RunStarted") == 1 and events.count("wfminiapi.RunEnded") == 1
    assert (tmp_path / "telemetry" / f"{report['session_id']}.wfminiapi.report.json").exists()


def test_failed_task_is_counted(tmp_path, backend):
    _, res = run("task_fail", tmp_path, backend)
    assert res["report"]["tasks"]["failed"] == 1
    assert res["report"]["telemetry_complete"]


def test_failed_dependency_is_not_submitted(tmp_path, backend):
    _, res = run("dep_fail", tmp_path, backend)
    tasks = res["report"]["tasks"]
    assert (tasks["failed"], tasks["not_submitted"]) == (1, 1)


def test_exception_is_reraised_without_waiting(tmp_path, backend):
    proc, res = run("raise", tmp_path, backend)
    assert proc.returncode == 1 and "ValueError: core raised" in proc.stderr
    assert res["report"]["telemetry_complete"]
    assert res["elapsed"] < 15  # no shutdown_wait_s (60 s) on a plain exception


@pytest.mark.parametrize("scenario", ["sigint", "sigterm"])
def test_signal_reaches_the_ordered_exit(scenario, tmp_path, backend):
    proc, res = run(scenario, tmp_path, backend)
    assert proc.returncode == 1  # AsyncFlow's own handler: CancelledError, as before
    assert res["report"]["interrupted"] and res["report"]["telemetry_complete"]
    events = checkpoint_events(tmp_path)
    assert "wfminiapi.RunEnded" in events and "SessionEnded" in events


def test_early_stop_is_reported(tmp_path, backend):
    _, res = run("early_stop", tmp_path, backend)
    assert not res["report"]["telemetry_complete"]
    assert any("stopped_before_exit" in r for r in res["report"]["reasons"])
    assert res["elapsed"] < 15


@pytest.mark.parametrize(
    "scenario,needle",
    [
        ("late_attach", "already submitted"),
        ("wrong_telemetry", "is not this flow's"),
    ],
)
def test_attach_errors(scenario, needle, tmp_path, backend):
    _, res = run(scenario, tmp_path, backend)
    assert res["attach_error"] and needle in res["attach_error"]


def test_miniapp_does_not_draw_from_global_random(tmp_path, backend):
    proc, res = run("random_state", tmp_path, backend)
    assert res["random_unchanged_by_enter"] and res["random_unchanged_by_exit"]
    if backend == "concurrent":  # the Dragon runtime itself draws from `random`
        assert "RANDOM_UNCHANGED True" in proc.stdout
