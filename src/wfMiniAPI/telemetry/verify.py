"""Verify a RHAPSODY telemetry checkpoint written under a MiniApp.

``verify_checkpoint`` is a pure function over the checkpoint file, shared by
``MiniApp``, CI and the equivalence harness.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .events import PREFIX

_TERMINAL = ("TaskCompleted", "TaskFailed", "TaskCanceled")
_BLOCK_PREFIX = "block."  # AsyncFlow's only block marker today (uid prefix)


@dataclass(frozen=True)
class TaskCounts:
    created: int = 0
    completed: int = 0
    failed: int = 0
    canceled: int = 0
    abandoned: int = 0  # submitted, no terminal event (e.g. cancelled at shutdown, UP-12)
    not_submitted: int = 0  # created, never submitted (e.g. a failed dependency)


@dataclass(frozen=True)
class MiniAppReport:
    session_id: str
    checkpoint: str | None
    telemetry_complete: bool
    ended_by: str = ""  # normal | exception | cancelled | engine_shutdown
    tasks: TaskCounts = field(default_factory=TaskCounts)
    events: dict[str, int] = field(default_factory=dict)  # wfminiapi events written, by type
    reasons: tuple[str, ...] = ()  # why telemetry is incomplete
    notes: tuple[str, ...] = ()  # worth knowing, but telemetry is still complete
    # a run of many engines and components (see run.py); empty for a run of one engine
    run_id: str = ""  # the run's engine session id
    part: str = ""  # which part of the run this engine is; "" for the run's engine
    parameters: dict = field(default_factory=dict)  # the run's configuration
    completed_by_executable: dict[str, int] = field(default_factory=dict)
    parts: dict = field(default_factory=dict)  # part name -> its engine's report
    components: dict = field(default_factory=dict)  # component name -> its verifier's verdict
    checks: tuple = ()  # each {name, ok, detail, verify}
    checks_pass: bool = True  # the results' validity, apart from telemetry completeness

    @property
    def exit_hint(self) -> int:
        """0 if telemetry is complete and the checks pass, 3 if telemetry is incomplete,
        4 if it is complete but a check failed. wfMiniAPI never exits by itself."""
        if not self.telemetry_complete:
            return 3
        return 0 if self.checks_pass else 4

    def to_dict(self) -> dict:
        return {**asdict(self), "exit_hint": self.exit_hint}


def find_checkpoint(directory: str | Path, session_id: str) -> tuple[Path | None, str | None]:
    matches = sorted(Path(directory).glob(f"{session_id}.*.telemetry.jsonl"))
    if len(matches) == 1:
        return matches[0], None
    return None, f"expected one checkpoint {session_id}.*.telemetry.jsonl, found {len(matches)}"


def _events(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            # cheap prefilter: resource/metric/span lines are most of the file
            if '"event_type": "Task' not in line and f'"event_type": "{PREFIX}' not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("section") == "event":
                yield rec


def verify_checkpoint(
    directory: str | Path,
    session_id: str,
    *,
    emitted: dict[str, int] | None = None,
    seen: dict[str, int] | None = None,
    ended_by: str = "",
    reasons: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> MiniAppReport:
    """Check that the session's checkpoint holds what was emitted, and count task outcomes.

    ``emitted`` / ``seen`` are the wfminiapi events MiniApp emitted and its subscriber saw
    (``seen`` also covers every other event type).  Telemetry is complete when the
    checkpoint exists and every count matches; tasks without a terminal event are
    reported, not treated as telemetry loss.  ``notes`` are passed through: facts that
    do not make telemetry incomplete.
    """
    reasons = list(reasons)
    path, problem = find_checkpoint(directory, session_id)
    if path is None:
        reasons.append(problem)
        return MiniAppReport(
            session_id, None, False, ended_by, reasons=tuple(reasons), notes=tuple(notes)
        )

    written: Counter[str] = Counter()
    created, submitted, terminal, executable = set(), set(), {}, {}
    for rec in _events(path):
        etype, uid = rec.get("event_type", ""), rec.get("task_id")
        written[etype] += 1
        if not uid or uid.startswith(_BLOCK_PREFIX):
            continue
        if etype == "TaskCreated":
            created.add(uid)
            executable[uid] = (rec.get("attributes") or {}).get("executable") or "?"
        elif etype == "TaskSubmitted":
            submitted.add(uid)
        elif etype in _TERMINAL:
            terminal.setdefault(uid, etype)  # first terminal state wins

    ended = Counter(terminal[u] for u in created if u in terminal)
    by_executable = Counter(executable[u] for u in created if terminal.get(u) == "TaskCompleted")
    open_tasks = created - terminal.keys()
    tasks = TaskCounts(
        created=len(created),
        completed=ended["TaskCompleted"],
        failed=ended["TaskFailed"],
        canceled=ended["TaskCanceled"],
        abandoned=len(open_tasks & submitted),
        not_submitted=len(open_tasks - submitted),
    )

    for etype, n in sorted((emitted or {}).items()):
        if written[etype] != n:
            reasons.append(f"{etype}: emitted {n}, written {written[etype]}")
    for etype, n in sorted((seen or {}).items()):
        if (etype.startswith(PREFIX) or etype.startswith("Task")) and written[etype] != n:
            reasons.append(f"{etype}: seen {n}, written {written[etype]}")

    ours = {k: v for k, v in written.items() if k.startswith(PREFIX)}
    return MiniAppReport(
        session_id=session_id,
        checkpoint=str(path),
        telemetry_complete=not reasons,
        ended_by=ended_by,
        tasks=tasks,
        events=dict(sorted(ours.items())),
        reasons=tuple(reasons),
        notes=tuple(notes),
        completed_by_executable=dict(sorted(by_executable.items())),
    )
