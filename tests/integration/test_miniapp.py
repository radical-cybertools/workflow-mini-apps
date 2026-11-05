"""MiniApp behaviour on the Concurrent backend; each scenario runs in its own process."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCENARIOS = Path(__file__).resolve().parent / "scenarios.py"


def run(scenario, tmp_path):
    proc = subprocess.run(
        [sys.executable, str(SCENARIOS), scenario, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
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


def test_ok_run_is_complete_and_blocks_are_not_tasks(tmp_path):
    proc, res = run("ok", tmp_path)
    assert proc.returncode == 0
    report = res["report"]
    assert report["telemetry_complete"], report["reasons"]
    assert report["tasks"]["created"] == 3 == report["tasks"]["completed"]
    events = checkpoint_events(tmp_path)
    assert events.count("wfminiapi.RunStarted") == 1 and events.count("wfminiapi.RunEnded") == 1
    assert (tmp_path / "telemetry" / f"{report['session_id']}.wfminiapi.report.json").exists()


def test_failed_task_is_counted(tmp_path):
    _, res = run("task_fail", tmp_path)
    assert res["report"]["tasks"]["failed"] == 1
    assert res["report"]["telemetry_complete"]


def test_failed_dependency_is_not_submitted(tmp_path):
    _, res = run("dep_fail", tmp_path)
    tasks = res["report"]["tasks"]
    assert (tasks["failed"], tasks["not_submitted"]) == (1, 1)


def test_exception_is_reraised_without_waiting(tmp_path):
    proc, res = run("raise", tmp_path)
    assert proc.returncode == 1 and "ValueError: core raised" in proc.stderr
    assert res["report"]["telemetry_complete"]
    assert res["elapsed"] < 15  # no shutdown_wait_s (60 s) on a plain exception


@pytest.mark.parametrize("scenario", ["sigint", "sigterm"])
def test_signal_reaches_the_ordered_exit(scenario, tmp_path):
    proc, res = run(scenario, tmp_path)
    assert proc.returncode == 1  # AsyncFlow's own handler: CancelledError, as before
    assert res["report"]["interrupted"] and res["report"]["telemetry_complete"]
    events = checkpoint_events(tmp_path)
    assert "wfminiapi.RunEnded" in events and "SessionEnded" in events


def test_early_stop_is_reported(tmp_path):
    _, res = run("early_stop", tmp_path)
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
def test_attach_errors(scenario, needle, tmp_path):
    _, res = run(scenario, tmp_path)
    assert res["attach_error"] and needle in res["attach_error"]


def test_miniapp_does_not_draw_from_global_random(tmp_path):
    proc, res = run("random_state", tmp_path)
    assert res["random_unchanged_in_block"]
    assert "RANDOM_UNCHANGED True" in proc.stdout
