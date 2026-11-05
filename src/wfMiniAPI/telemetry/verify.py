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
    interrupted: bool = False
    tasks: TaskCounts = field(default_factory=TaskCounts)
    events: dict[str, int] = field(default_factory=dict)  # wfminiapi events written, by type
    reasons: tuple[str, ...] = ()

    @property
    def exit_hint(self) -> int:
        """0 if telemetry is complete, 3 if not. wfMiniAPI never exits by itself."""
        return 0 if self.telemetry_complete else 3

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
    interrupted: bool = False,
    reasons: tuple[str, ...] = (),
) -> MiniAppReport:
    """Check that the session's checkpoint holds what was emitted, and count task outcomes.

    ``emitted`` / ``seen`` are the wfminiapi events MiniApp emitted and its subscriber saw
    (``seen`` also covers every other event type).  Telemetry is complete when the
    checkpoint exists and every count matches; tasks without a terminal event are
    reported, not treated as telemetry loss.
    """
    reasons = list(reasons)
    path, problem = find_checkpoint(directory, session_id)
    if path is None:
        reasons.append(problem)
        return MiniAppReport(session_id, None, False, interrupted, reasons=tuple(reasons))

    written: Counter[str] = Counter()
    created, submitted, terminal = set(), set(), {}
    for rec in _events(path):
        etype, uid = rec.get("event_type", ""), rec.get("task_id")
        written[etype] += 1
        if not uid or uid.startswith(_BLOCK_PREFIX):
            continue
        if etype == "TaskCreated":
            created.add(uid)
        elif etype == "TaskSubmitted":
            submitted.add(uid)
        elif etype in _TERMINAL:
            terminal.setdefault(uid, etype)  # first terminal state wins

    ended = Counter(terminal[u] for u in created if u in terminal)
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
        interrupted=interrupted,
        tasks=tasks,
        events=dict(sorted(ours.items())),
        reasons=tuple(reasons),
    )
