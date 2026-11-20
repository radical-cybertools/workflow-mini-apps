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
    assert report["ended_by"] == "normal"
    assert report["tasks"]["created"] == 3 == report["tasks"]["completed"]
    # the exit runs in the caller's Context, so RHAPSODY's OTel context detach works
    assert "Failed to detach context" not in proc.stderr
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
    assert res["report"]["ended_by"] == "exception"


@pytest.mark.parametrize("scenario", ["sigint", "sigterm"])
def test_signal_reaches_the_ordered_exit(scenario, tmp_path, backend):
    proc, res = run(scenario, tmp_path, backend)
    assert proc.returncode == 1  # AsyncFlow's own handler: CancelledError, as before
    assert res["report"]["ended_by"] == "engine_shutdown"
    assert res["report"]["telemetry_complete"], res["report"]["reasons"]
    events = checkpoint_events(tmp_path)
    assert "wfminiapi.RunEnded" in events and "SessionEnded" in events


def test_early_stop_is_reported(tmp_path, backend):
    _, res = run("early_stop", tmp_path, backend)
    assert not res["report"]["telemetry_complete"]
    assert any("stopped_before_exit" in r for r in res["report"]["reasons"])


@pytest.mark.parametrize(
    "scenario,needle",
    [
        ("late_attach", "already submitted"),
        ("wrong_telemetry", "is not this flow's"),
        ("hand_built_no_start", "this flow has no telemetry"),
        ("hand_built_telemetry", "not the manager this flow emits into"),
        ("second_start_telemetry", "not the manager this flow emits into"),
    ],
)
def test_attach_errors(scenario, needle, tmp_path, backend):
    if scenario == "second_start_telemetry" and backend == "dragon":
        # Upstream, not MiniApp: with two telemetry managers on one Dragon backend, the
        # process hangs at exit about 1 run in 10 (RHAPSODY's Dragon adapter leaves its
        # dragon.native Queue to the garbage collector, whose __del__ blocks once the
        # runtime is gone). The attach check itself does not depend on the backend.
        pytest.skip("two telemetry managers on Dragon can hang at exit (upstream)")
    _, res = run(scenario, tmp_path, backend)
    assert res["attach_error"] and needle in res["attach_error"]
    # the script's finally still verifies: the report says MiniApp did not attach
    assert res["report"]["reasons"][0].startswith("not attached: AttachError")


@pytest.mark.parametrize("scenario", ["outer_cancel", "ddsim_outer_cancel"])
def test_outer_cancel_is_not_a_signal(scenario, tmp_path, backend):
    # F1: a timeout around the run, also when the core's own finally shut the engine
    # down, is a cancel from outside: no wait for a shutdown nobody started
    proc, res = run(scenario, tmp_path, backend)
    assert proc.returncode == 1 and res["raised"] == "TimeoutError"
    assert res["report"]["ended_by"] == "cancelled"
    assert res["report"]["telemetry_complete"], res["report"]["reasons"]
    if backend == "concurrent":
        assert res["elapsed"] < 20


def test_a_kept_backend_stays_running(tmp_path, backend):
    # F10: shutdown(skip_execution_backend=True) in the block keeps the backend
    proc, res = run("kept_backend", tmp_path, backend)
    assert proc.returncode == 0
    assert res["report"]["ended_by"] == "normal"
    assert res["report"]["telemetry_complete"], res["report"]["reasons"]
    assert res["backend_state"] != "SHUTDOWN"
    assert res["backend_state_after_owner"] == "SHUTDOWN"


def test_miniapp_does_not_draw_from_global_random(tmp_path, backend):
    proc, res = run("random_state", tmp_path, backend)
    assert res["random_unchanged_by_enter"] and res["random_unchanged_by_exit"]
    if backend == "concurrent":  # the Dragon runtime itself draws from `random`
        assert "RANDOM_UNCHANGED True" in proc.stdout


def test_a_core_that_still_shuts_down_is_complete(tmp_path, backend):
    _, res = run("core_shuts_down", tmp_path, backend)
    assert res["report"]["ended_by"] == "normal"
    assert res["report"]["telemetry_complete"], res["report"]["reasons"]
    assert res["report"]["events"]["wfminiapi.RunEnded"] == 1


def test_telemetry_stopped_by_the_engine_shutdown_is_a_note(tmp_path, backend):
    # what AsyncFlow #125's shutdown() will do: RunEnded cannot reach the checkpoint,
    # but no task event is lost
    _, res = run("engine_stopped_telemetry", tmp_path, backend)
    report = res["report"]
    assert report["telemetry_complete"], report["reasons"]
    assert "engine shutdown stopped telemetry" in report["notes"][0]
    assert "wfminiapi.RunEnded" not in report["events"]


def test_a_stop_by_hand_before_the_shutdown_is_incomplete(tmp_path, backend):
    _, res = run("hand_stop_then_shutdown", tmp_path, backend)
    assert not res["report"]["telemetry_complete"]
    assert res["report"]["reasons"][0].startswith("stopped_before_exit")


def test_verify_too_early_reports_and_does_not_raise(tmp_path, backend):
    proc, res = run("verify_too_early", tmp_path, backend)
    assert proc.returncode == 0
    assert res["early_report"]["reasons"] == [
        "telemetry still running: call verify() after flow.shutdown() and telemetry.stop()"
    ]
    assert res["report"]["telemetry_complete"], res["report"]["reasons"]


def test_a_failed_shutdown_still_stops_telemetry_and_writes_the_report(tmp_path, backend):
    # the nested finally: stop() and verify() run although flow.shutdown() raised
    proc, res = run("shutdown_fails", tmp_path, backend)
    assert proc.returncode == 1 and res["raised"] == "RuntimeError"
    assert res["report"]["telemetry_complete"], res["report"]["reasons"]
    assert "SessionEnded" in checkpoint_events(tmp_path)


def test_a_script_that_skips_verify_is_warned_at_exit(tmp_path, backend):
    proc, res = run("no_verify", tmp_path, backend)
    assert proc.returncode == 0 and res["report"] is None
    assert "verify() was never called" in proc.stderr
