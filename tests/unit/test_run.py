"""A run of many engines and components: parts, component verifiers, checks, offline."""

import json

import pytest

from wfMiniAPI.telemetry.run import (
    ComponentReport,
    PartRef,
    Run,
    collect_parts,
    resolve,
    run_checks,
    verify_components,
    verify_run,
    write_manifest,
    write_part,
)
from wfMiniAPI.telemetry.verify import MiniAppReport, verify_checkpoint

SID = "asyncflow.session.test"


def event(etype, task_id=None, **attributes):
    return {
        "event_type": etype,
        "event_time": 0.0,
        "session_id": SID,
        "task_id": task_id,
        "attributes": attributes,
        "section": "event",
    }


def write(directory, records, session=SID):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session}.1.telemetry.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def a_task(uid, executable, outcome="TaskCompleted"):
    return [
        event("TaskCreated", uid, executable=executable),
        event("TaskSubmitted", uid),
        event(outcome, uid),
    ]


def report(**fields):
    base = {"session_id": "asyncflow.session.p", "checkpoint": "x", "telemetry_complete": True}
    return MiniAppReport(**{**base, **fields}).to_dict()


def test_completed_tasks_are_counted_by_executable(tmp_path):
    write(
        tmp_path,
        a_task("task.000001", "sim")
        + a_task("task.000002", "sim")
        + a_task("task.000003", "rank")
        + a_task("task.000004", "sim", outcome="TaskFailed"),
    )
    r = verify_checkpoint(tmp_path, SID)
    assert r.completed_by_executable == {"rank": 1, "sim": 2}
    assert r.tasks.failed == 1


def test_a_part_s_report_is_collected(tmp_path):
    ref = PartRef(run_id="run.1", name="p0", parts_dir=str(tmp_path / "parts"))
    write_part(ref, report(run_id="run.1", part="p0"))
    parts, reasons = collect_parts(tmp_path / "parts", "run.1", ["p0"])
    assert not reasons
    assert parts["p0"]["telemetry_complete"]


def test_a_missing_incomplete_or_foreign_part_makes_the_run_incomplete(tmp_path):
    parts_dir = tmp_path / "parts"
    for name, fields in {
        "p0": {"run_id": "run.1", "part": "p0"},
        "p1": {"run_id": "run.1", "part": "p1", "telemetry_complete": False, "reasons": ("lost",)},
        "p3": {"run_id": "run.9", "part": "p3"},
    }.items():
        write_part(PartRef(fields["run_id"], name, str(parts_dir)), report(**fields))
    parts, reasons = collect_parts(parts_dir, "run.1", ["p0", "p1", "p2", "p3"])
    assert parts["p0"]["telemetry_complete"]
    assert any(r.startswith("part p1:") and "lost" in r for r in reasons), reasons
    assert any(r.startswith("part p2:") and "no report" in r for r in reasons), reasons
    assert any(r.startswith("part p3:") and "run.9" in r for r in reasons), reasons
    assert len(reasons) == 3


def test_resolve_names_a_callable_by_its_import_path():
    path, fn = resolve("json:dumps")
    assert path == "json:dumps" and fn is json.dumps
    path, fn = resolve(json.loads)
    assert path == "json:loads" and fn is json.loads
    path, _ = resolve(lambda files: True)
    assert path == ""  # not importable: verified in the run, not again offline
    with pytest.raises(ValueError):
        resolve("no_colon_here")


def test_component_verifiers(tmp_path):
    def good(files):
        return ComponentReport(True, counts={"rows": 3})

    def bad(files):
        return ComponentReport(False, reasons=("export cut short",))

    def boom(files):
        raise ValueError("unreadable")

    components = {name: (str(tmp_path), fn) for name, fn in {"good": good, "bad": bad}.items()}
    components["boom"] = (str(tmp_path), boom)
    components["bool"] = (str(tmp_path), lambda files: True)
    summaries, reasons = verify_components(components)
    assert summaries["good"]["complete"] and summaries["good"]["counts"] == {"rows": 3}
    assert summaries["bool"]["complete"]
    assert not summaries["bad"]["complete"] and not summaries["boom"]["complete"]
    assert "component bad: export cut short" in reasons
    assert any(r.startswith("component boom: ValueError: unreadable") for r in reasons)
    assert len(reasons) == 2


def test_checks_pass_fail_raise_or_are_recorded(tmp_path):
    run = Run(run_id="run.1", parameters={"agents": 2}, parts={}, components={}, report={})
    checks = [
        {"name": "passes", "target": lambda r: r.parameters["agents"] == 2},
        {"name": "with_detail", "target": lambda r: (False, {"agents": r.parameters["agents"]})},
        {"name": "raises", "target": lambda r: 1 / 0},
        {"name": "static", "ok": True, "detail": "computed in the run"},
    ]
    results = {c["name"]: c for c in run_checks(checks, run)}
    assert results["passes"]["ok"] is True
    assert results["with_detail"] == {
        "name": "with_detail",
        "ok": False,
        "detail": {"agents": 2},
        "verify": "",
    }
    assert results["raises"]["ok"] is False
    assert "ZeroDivisionError" in results["raises"]["detail"]
    assert results["static"]["ok"] is True and results["static"]["detail"] == "computed in the run"


def test_exit_hint_separates_failed_checks_from_incomplete_telemetry():
    ok = {"session_id": "s", "checkpoint": "c"}
    assert MiniAppReport(**ok, telemetry_complete=True).exit_hint == 0
    assert MiniAppReport(**ok, telemetry_complete=True, checks_pass=False).exit_hint == 4
    assert MiniAppReport(**ok, telemetry_complete=False, checks_pass=False).exit_hint == 3


def test_a_run_verifies_again_offline_from_its_files(tmp_path, monkeypatch):
    # a verifier and a check importable by name, as a component or application would ship
    (tmp_path / "offline_helpers.py").write_text(
        "from pathlib import Path\n"
        "from wfMiniAPI.telemetry.run import ComponentReport\n"
        "def verify_rows(files):\n"
        "    n = len(Path(files, 'rows.txt').read_text().split())\n"
        "    return ComponentReport(True, counts={'rows': n})\n"
        "def rows_match_sims(run):\n"
        "    sims = sum(p['completed_by_executable'].get('sim', 0) for p in run.parts.values())\n"
        "    return sims == run.components['rows']['counts']['rows'], sims\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    launcher, parts_dir = tmp_path / "launcher", tmp_path / "launcher" / f"{SID}.parts"
    write(launcher, [event("wfminiapi.RunStarted"), event("wfminiapi.RunEnded")])
    write_part(
        PartRef(SID, "p0", str(parts_dir)),
        report(run_id=SID, part="p0", completed_by_executable={"sim": 2}),
    )
    (tmp_path / "rows").mkdir()
    (tmp_path / "rows" / "rows.txt").write_text("a b\n")
    manifest = {
        "schema": "wfminiapi/1",
        "run_id": SID,
        "session_id": SID,
        "checkpoint_dir": str(launcher),
        "parameters": {"agents": 2},
        "parts": ["p0"],
        "parts_dir": str(parts_dir),
        "components": {
            "rows": {"files": str(tmp_path / "rows"), "verify": "offline_helpers:verify_rows"}
        },
        "checks": [
            {"name": "rows_match_sims", "verify": "offline_helpers:rows_match_sims"},
            {"name": "static", "verify": "", "ok": True, "detail": "from the run"},
        ],
    }
    write_manifest(launcher / f"{SID}.wfminiapi.manifest.json", manifest)

    r = verify_run(launcher)
    assert r.telemetry_complete, r.reasons
    assert r.checks_pass and r.exit_hint == 0
    assert r.parameters == {"agents": 2}
    assert set(r.parts) == {"p0"} and r.components["rows"]["complete"]
    assert [c["name"] for c in r.checks] == ["rows_match_sims", "static"]

    (tmp_path / "rows" / "rows.txt").write_text("a b c\n")  # the files changed: the check fails
    r = verify_run(launcher)
    assert r.telemetry_complete and not r.checks_pass and r.exit_hint == 4

    (parts_dir / "p0.json").unlink()  # a part's report is gone: incomplete
    r = verify_run(launcher)
    assert not r.telemetry_complete and r.exit_hint == 3
