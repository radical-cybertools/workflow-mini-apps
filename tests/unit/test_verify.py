"""verify_checkpoint against small checkpoint fixtures."""

import json

from wfMiniAPI.telemetry.verify import verify_checkpoint

SID = "asyncflow.session.test"


def event(etype, task_id=None):
    return {
        "event_id": "x",
        "event_type": etype,
        "event_time": 0.0,
        "emit_time": 0.0,
        "session_id": SID,
        "backend": "b",
        "task_id": task_id,
        "node_id": None,
        "attributes": {},
        "trace_id": None,
        "span_id": None,
        "name": etype,
        "section": "event",
    }


def write(tmp_path, records, name=f"{SID}.1.telemetry.jsonl"):
    lines = [json.dumps(r) for r in records]
    lines.insert(1, json.dumps({"section": "metric", "name": "tasks_running"}))
    (tmp_path / name).write_text("\n".join(lines) + "\n")


def test_complete_with_task_outcomes(tmp_path):
    write(
        tmp_path,
        [
            event("wfminiapi.RunStarted"),
            event("TaskCreated", "task.000001"),
            event("TaskSubmitted", "task.000001"),
            event("TaskCompleted", "task.000001"),
            event("TaskCreated", "block.000001"),  # blocks are not tasks
            event("TaskCreated", "task.000002"),
            event("TaskSubmitted", "task.000002"),
            event("TaskFailed", "task.000002"),
            event("TaskCreated", "task.000003"),
            event("TaskSubmitted", "task.000003"),  # abandoned
            event("TaskCreated", "task.000004"),  # never submitted
            event("wfminiapi.RunEnded"),
        ],
    )
    report = verify_checkpoint(
        tmp_path, SID, emitted={"wfminiapi.RunStarted": 1, "wfminiapi.RunEnded": 1}
    )
    assert report.telemetry_complete, report.reasons
    t = report.tasks
    assert (t.created, t.completed, t.failed, t.abandoned, t.not_submitted) == (4, 1, 1, 1, 1)
    assert report.exit_hint == 0


def test_lost_event_is_incomplete(tmp_path):
    write(tmp_path, [event("wfminiapi.RunStarted")])
    report = verify_checkpoint(
        tmp_path, SID, emitted={"wfminiapi.RunStarted": 1, "wfminiapi.RunEnded": 1}
    )
    assert not report.telemetry_complete and report.exit_hint == 3
    assert any("RunEnded" in r for r in report.reasons)


def test_seen_but_not_written_is_incomplete(tmp_path):
    write(tmp_path, [event("TaskCreated", "task.000001")])
    report = verify_checkpoint(tmp_path, SID, seen={"TaskCreated": 2})
    assert not report.telemetry_complete


def test_needs_exactly_one_checkpoint(tmp_path):
    assert not verify_checkpoint(tmp_path, SID).telemetry_complete
    write(tmp_path, [], name=f"{SID}.1.telemetry.jsonl")
    write(tmp_path, [], name=f"{SID}.2.telemetry.jsonl")
    report = verify_checkpoint(tmp_path, SID)
    assert not report.telemetry_complete and "found 2" in report.reasons[0]
