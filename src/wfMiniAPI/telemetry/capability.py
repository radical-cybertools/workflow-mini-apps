"""The telemetry capability of a MiniApp: attaches to the user's AsyncFlow telemetry.

It never creates, configures or replaces the user's components.  It reads two things
RHAPSODY does not expose publicly yet (marked ``PRIVATE``; covered by drift tests):
the checkpoint directory and the running flag.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections import Counter
from importlib import metadata
from pathlib import Path

from rhapsody.telemetry.events import make_event

from .events import PREFIX, Barrier, RunEnded, RunStarted
from .verify import MiniAppReport, find_checkpoint, verify_checkpoint

log = logging.getLogger("wfMiniAPI.telemetry")

BARRIER_TIMEOUT_S = 30.0


class AttachError(RuntimeError):
    """MiniApp cannot attach: its telemetry would be wrong or incomplete."""


def _package(dist: str) -> str:
    """'<version>' or '<version>@<commit>' from installed package metadata."""
    try:
        d = metadata.distribution(dist)
    except metadata.PackageNotFoundError:
        return "not installed"
    commit = ""
    try:
        info = json.loads(d.read_text("direct_url.json") or "{}")
        commit = info.get("vcs_info", {}).get("commit_id", "")[:7]
    except ValueError:
        pass
    return f"{d.version}@{commit}" if commit else d.version


class MiniAppTelemetry:
    """Attach-time checks, run events, the ordered stop and the end-of-run verification."""

    def __init__(self, flow, telemetry) -> None:
        self.flow = flow
        self.manager = telemetry  # the user's RHAPSODY TelemetryManager, as passed in
        self.checkpoint_dir: Path | None = None
        self.report: MiniAppReport | None = None
        self._emitted: Counter[str] = Counter()
        self._seen: Counter[str] = Counter()
        self._barrier_seen: asyncio.Event | None = None
        self._t0 = 0.0

    # ------------------------------------------------------------------ attach
    def check(self) -> None:
        """Raise AttachError unless telemetry belongs to this flow and can be verified."""
        if self.manager is None:
            raise AttachError(
                "telemetry is None: call `telemetry = await flow.start_telemetry(...)`"
            )
        if self.manager.session_id != self.flow.uid:
            raise AttachError(
                f"telemetry session {self.manager.session_id!r} is not this flow's "
                f"({self.flow.uid!r}): pass the manager returned by this flow's start_telemetry()"
            )
        if not getattr(self.manager, "_running", False):  # PRIVATE (drift-tested)
            raise AttachError("telemetry is not running: start it before entering MiniApp")
        checkpoint = getattr(self.manager, "_checkpoint_path", None)  # PRIVATE (drift-tested)
        if not checkpoint:
            raise AttachError(
                "telemetry has no checkpoint file to verify: "
                "use flow.start_telemetry(checkpoint_path=...)"
            )
        self.checkpoint_dir = Path(checkpoint)
        path, problem = find_checkpoint(self.checkpoint_dir, self.manager.session_id)
        if path is None:
            raise AttachError(f"telemetry checkpoint not found: {problem}")
        submitted = sorted(self.flow.running | self.flow.resolved)
        if submitted:
            raise AttachError(
                f"tasks were already submitted before MiniApp attached ({submitted[:5]}...): "
                "enter MiniApp right after start_telemetry()"
            )
        if "drain_on_stop" not in inspect.signature(type(self.manager).__init__).parameters:
            raise AttachError(
                "RHAPSODY telemetry lacks drain_on_stop: install RHAPSODY main (142f396 or newer)"
            )

    async def start(self, backends) -> None:
        self._t0 = time.time()
        self._barrier_seen = asyncio.Event()
        self.manager.subscribe(self._on_event)
        if not await self._barrier("attach"):
            raise AttachError("telemetry dispatch loop did not respond: is it running?")
        self._emit(
            RunStarted,
            wfminiapi_version=_package("wfMiniAPI"),
            asyncflow_version=_package("radical.asyncflow"),
            rhapsody_version=_package("rhapsody-py"),
            backends=",".join(type(b).__name__ for b in backends),
        )

    # ------------------------------------------------------------------ events
    def _on_event(self, event) -> None:  # O(1): runs in RHAPSODY's dispatch loop
        self._seen[event.event_type] += 1
        if event.event_type == Barrier.__name__ and self._barrier_seen is not None:
            self._barrier_seen.set()

    def _emit(self, cls, **fields) -> None:
        self.manager.emit(
            make_event(cls, session_id=self.manager.session_id, backend="wfminiapi", **fields)
        )
        self._emitted[cls.__name__] += 1

    async def _barrier(self, purpose: str) -> bool:
        """Wait until every event emitted so far has been processed (the queue is FIFO)."""
        self._barrier_seen.clear()
        self._emit(Barrier, purpose=purpose)
        try:
            await asyncio.wait_for(self._barrier_seen.wait(), BARRIER_TIMEOUT_S)
            return True
        except asyncio.TimeoutError:
            return False

    # ------------------------------------------------------------------ exit
    async def finish(self, *, interrupted: bool, exception_type: str, reasons: list[str]) -> None:
        """Run-end events, stop telemetry (always), verify the checkpoint."""
        try:
            # PRIVATE read (drift-tested): a stopped manager drops emits silently (UP-08),
            # so don't wait for a barrier that can never arrive
            if not getattr(self.manager, "_running", False) or not await self._barrier("exit"):
                reasons.append("stopped_before_exit: telemetry stopped before MiniApp exited")
            else:
                self._emit(
                    RunEnded,
                    interrupted=interrupted,
                    exception_type=exception_type,
                    duration_s=time.time() - self._t0,
                )
        except Exception as exc:  # never mask the workflow's own outcome
            reasons.append(f"run-end events failed: {type(exc).__name__}: {exc}")
        finally:
            try:
                await self.manager.stop()
            except Exception as exc:
                reasons.append(f"telemetry.stop() failed: {type(exc).__name__}: {exc}")
        self.report = verify_checkpoint(
            self.checkpoint_dir,
            self.manager.session_id,
            emitted={k: v for k, v in self._emitted.items() if k.startswith(PREFIX)},
            seen=dict(self._seen),
            interrupted=interrupted,
            reasons=tuple(reasons),
        )
        self._write_report()
        self._log_report()

    def _write_report(self) -> None:
        try:
            out = self.checkpoint_dir / f"{self.manager.session_id}.wfminiapi.report.json"
            out.write_text(json.dumps(self.report.to_dict(), indent=1) + "\n")
        except OSError as exc:
            log.warning("could not write the MiniApp report: %s", exc)

    def _log_report(self) -> None:
        r, t = self.report, self.report.tasks
        msg = (
            f"MiniApp {r.session_id}: telemetry "
            f"{'complete' if r.telemetry_complete else 'INCOMPLETE'}; tasks created={t.created} "
            f"completed={t.completed} failed={t.failed} canceled={t.canceled} "
            f"abandoned={t.abandoned} not_submitted={t.not_submitted}"
        )
        if r.reasons:
            msg += "; " + "; ".join(r.reasons)
        log.log(logging.INFO if r.telemetry_complete else logging.WARNING, msg)
