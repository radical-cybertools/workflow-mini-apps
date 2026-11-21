"""A run of many engines: parts, a component and checks, on the Concurrent backend.

Each scenario runs in its own process (``parts_scenarios.py``).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENARIOS = HERE / "parts_scenarios.py"


def run(scenario, tmp_path):
    proc = subprocess.run(
        [sys.executable, str(SCENARIOS), scenario, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc, json.loads(lines[-1][len("RESULT ") :])


def events(directory, event_type):
    (path,) = Path(directory).glob("*.telemetry.jsonl")
    records = (json.loads(ln) for ln in path.read_text().splitlines())
    return [r for r in records if r.get("event_type") == event_type]


def offline(tmp_path, cwd=None, *args):
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")])
    )
    return subprocess.run(
        [sys.executable, "-m", "wfMiniAPI.verify", str(tmp_path / "launcher"), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        cwd=cwd,
    )


def test_parts_a_component_and_a_check_make_one_verified_run(tmp_path):
    proc, res = run("parts_ok", tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = res["report"]
    assert report["telemetry_complete"], report["reasons"]
    assert report["checks_pass"] and report["exit_hint"] == 0
    assert report["run_id"] == res["run_id"] and report["part"] == ""
    assert report["parameters"] == {"parts": 3, "sims_per_part": 3}
    assert set(report["parts"]) == {"p0", "p1", "p2"}
    for name, part in report["parts"].items():
        assert part["telemetry_complete"], (name, part["reasons"])
        assert part["completed_by_executable"] == {"sim": 3}
    assert report["components"]["results"]["complete"]
    assert report["components"]["results"]["counts"] == {"results": 9}
    assert [(c["name"], c["ok"]) for c in report["checks"]] == [("sims_match_results", True)]

    # every part's run record names the run and the part, wherever the part ran
    for name in ("p0", "p1", "p2"):
        (started,) = events(tmp_path / "parts" / name, "wfminiapi.RunStarted")
        assert (started["run_id"], started["part"]) == (res["run_id"], name)
    # the parameters are in the run's own checkpoint, and the run left a manifest
    (params,) = events(tmp_path / "launcher", "wfminiapi.Parameters")
    assert json.loads(params["parameters"]) == {"parts": 3, "sims_per_part": 3}
    (manifest,) = (tmp_path / "launcher").glob("*.wfminiapi.manifest.json")
    assert json.loads(manifest.read_text())["parts"] == ["p0", "p1", "p2"]


def test_a_part_that_never_ran_makes_the_run_incomplete(tmp_path):
    proc, res = run("part_missing", tmp_path)
    report = res["report"]
    assert not report["telemetry_complete"]
    assert any(r.startswith("part p2:") for r in report["reasons"]), report["reasons"]
    assert report["exit_hint"] == 3


def test_a_failed_check_is_not_incomplete_telemetry(tmp_path):
    proc, res = run("check_fails", tmp_path)
    report = res["report"]
    assert report["telemetry_complete"], report["reasons"]
    assert not report["checks_pass"] and report["exit_hint"] == 4
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["sims_match_results"]["ok"] and not checks["always_fails"]["ok"]
    assert checks["always_fails"]["verify"] == "parts_helpers:always_fails"
    assert checks["static"] == {
        "name": "static",
        "ok": True,
        "detail": "computed in the run",
        "verify": "",
    }


def test_the_run_verifies_again_offline(tmp_path):
    proc, res = run("parts_ok", tmp_path)
    assert res["report"]["exit_hint"] == 0
    out = offline(tmp_path)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    again = json.loads(out.stdout)
    assert set(again["parts"]) == {"p0", "p1", "p2"}
    assert again["checks_pass"] and again["components"]["results"]["complete"]

    for f in (tmp_path / "launcher" / f"{res['run_id']}.parts").glob("p1.*.json"):
        f.unlink()
    out = offline(tmp_path)
    assert out.returncode == 3
    assert any(r.startswith("part p1:") for r in json.loads(out.stdout)["reasons"])


def test_relative_checkpoint_paths_verify_from_anywhere_and_after_a_move(tmp_path):
    work = tmp_path / "run"
    work.mkdir()
    proc, res = run("relative", work)  # the scenario's working directory is `work`
    assert res["report"]["exit_hint"] == 0, res["report"]["reasons"]
    out = offline(work, cwd="/")  # another working directory
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    moved = tmp_path / "moved"
    shutil.copytree(work, moved)
    shutil.rmtree(work)
    out = offline(moved, cwd="/")
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert set(json.loads(out.stdout)["parts"]) == {"p0", "p1", "p2"}


def test_details_and_counts_that_are_not_json_do_not_break_verify(tmp_path):
    proc, res = run("odd_details", tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = res["report"]
    assert report["exit_hint"] == 0, report["reasons"]
    odd = next(c for c in report["checks"] if c["name"] == "odd")
    assert odd["detail"]["('p0', 1)"] == 3
    assert report["components"]["odd"]["counts"]["('rank', 0)"] == 1
    (written,) = (tmp_path / "launcher").glob("*.wfminiapi.report.json")
    assert json.loads(written.read_text())["exit_hint"] == 0


def test_two_runs_in_one_checkpoint_directory_verify_by_session(tmp_path):
    _, first = run("parts_ok", tmp_path)
    _, second = run("parts_ok", tmp_path)
    out = offline(tmp_path)
    assert out.returncode == 3 and first["run_id"] in out.stdout  # which run? it says
    for res in (first, second):
        out = offline(tmp_path, None, "--session", res["run_id"])
        assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
