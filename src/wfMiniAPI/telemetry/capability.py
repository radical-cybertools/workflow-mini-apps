"""The telemetry capability of a MiniApp: attaches to the user's AsyncFlow telemetry.

It never creates, configures, replaces or stops the user's components: the script
shuts the engine down and stops telemetry, then calls ``verify()``.  From AsyncFlow it
uses the public ``flow.telemetry`` (the manager the engine emits into).  It reads two
things RHAPSODY does not expose publicly yet (marked ``PRIVATE``; covered by drift
tests): the checkpoint directory and the running flag.
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
from urllib.parse import unquote, urlparse

from rhapsody.telemetry.events import make_event

from .events import PREFIX, SCHEMA, Barrier, Parameters, RunEnded, RunStarted
from .run import MANIFEST_SUFFIX, PartRef, assemble, resolve, valid_name, write_manifest, write_part
from .verify import MiniAppReport, find_checkpoint, verify_checkpoint

log = logging.getLogger("wfMiniAPI.telemetry")

BARRIER_TIMEOUT_S = 30.0


class AttachError(RuntimeError):
    """MiniApp cannot attach: its telemetry would be wrong or incomplete."""


def _git_head(repo: Path) -> str:
    """Commit of a git checkout, read from its files (no subprocess); '' if unknown."""
    try:
        git = repo / ".git"
        if git.is_file():  # a worktree: "gitdir: <path>"
            git = (repo / git.read_text().split(":", 1)[1].strip()).resolve()
        common = git
        if (git / "commondir").is_file():  # refs of a worktree live in the main repository
            common = (git / (git / "commondir").read_text().strip()).resolve()
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head  # detached
        ref = head.split(":", 1)[1].strip()
        for base in (git, common):
            if (base / ref).is_file():
                return (base / ref).read_text().strip()
        for line in (common / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except (OSError, IndexError):
        pass
    return ""


def _package(dist: str) -> str:
    """'<version>' or '<version>@<commit>' from installed package metadata.

    The commit comes from the VCS URL of a git install, or from the checkout of an
    editable install (both main branches still declare the same version, UP-16).
    """
    try:
        d = metadata.distribution(dist)
    except metadata.PackageNotFoundError:
        return "not installed"
    commit = ""
    try:
        info = json.loads(d.read_text("direct_url.json") or "{}")
        commit = info.get("vcs_info", {}).get("commit_id", "")
        url = info.get("url", "")
        if not commit and info.get("dir_info", {}).get("editable") and url.startswith("file://"):
            commit = _git_head(Path(unquote(urlparse(url).path)))
    except ValueError:
        pass
    return f"{d.version}@{commit[:7]}" if commit else d.version


class MiniAppTelemetry:
    """Attach-time checks, run events and the end-of-run verification."""

    def __init__(self, flow, telemetry, *, part_of: PartRef | None = None) -> None:
        self.flow = flow
        self.manager = telemetry  # the user's RHAPSODY TelemetryManager, as passed in
        if part_of is not None and not isinstance(part_of, PartRef):
            raise TypeError("part_of: pass the PartRef that the run's app.part(name) returned")
        self.part_of = part_of
        # a run of many engines and components (run.py)
        self.parameters: dict = {}
        self.parts: list[str] = []
        self.components: dict[str, tuple[str, object]] = {}
        self.checks: list[dict] = []
        self._engine_reasons: tuple[str, ...] = ()
        self.checkpoint_dir: Path | None = None
        self.report: MiniAppReport | None = None
        self.attach_error: str | None = None
        self.started = False  # attach checks passed and RunStarted is queued
        self.ended_by = ""
        self.exception_type = ""
        self.reasons: list[str] = []
        self.notes: list[str] = []
        self._run_ended_queued = False
        # set when RHAPSODY's stop() sends SessionEnded, before it writes out its queue
        self._session_ended_seen = False
        self._engine_down_at_stop: bool | None = None
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
        engine = type(self.flow)
        if not hasattr(engine, "telemetry") or not hasattr(engine, "is_shutting_down"):
            raise AttachError(
                "this AsyncFlow has no joinable shutdown (flow.telemetry, "
                "flow.is_shutting_down): install the radical.asyncflow pinned by wfMiniAPI"
            )
        if self.manager.session_id != self.flow.uid:
            raise AttachError(
                f"telemetry session {self.manager.session_id!r} is not this flow's "
                f"({self.flow.uid!r}): pass the manager returned by this flow's start_telemetry()"
            )
        if self.flow.telemetry is None:
            raise AttachError(
                "this flow has no telemetry: call `telemetry = await flow.start_telemetry(...)`"
            )
        if self.manager is not self.flow.telemetry:
            raise AttachError(
                "telemetry is not the manager this flow emits into (built by hand, or from "
                "an earlier start_telemetry() call): pass the object the latest "
                "flow.start_telemetry() returned"
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
            run_id=self.run_id,
            part=self.part_of.name if self.part_of else "",
        )
        self.started = True
        if self.parameters:
            self._emit_parameters()

    # ------------------------------------------------------------------ the run
    @property
    def run_id(self) -> str:
        if self.part_of is not None:
            return self.part_of.run_id
        return str(getattr(self.manager, "session_id", "") or getattr(self.flow, "uid", ""))

    @property
    def parts_dir(self) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / f"{self.run_id}.parts"

    def set_parameters(self, values) -> None:
        self.parameters = json.loads(json.dumps(dict(values), default=str))
        if self.started and getattr(self.manager, "_running", False):  # PRIVATE (drift-tested)
            self._emit_parameters()

    def _emit_parameters(self) -> None:
        self._emit(Parameters, parameters=json.dumps(self.parameters, sort_keys=True))

    def add_part(self, name: str) -> PartRef:
        valid_name(name, "part")
        if self.part_of is not None:
            raise RuntimeError("a part cannot hand out parts: call part() on the run's MiniApp")
        if self.parts_dir is None:
            raise RuntimeError("enter the MiniApp (async with app) before handing out parts")
        if name in self.parts:
            raise ValueError(f"part {name!r} was already handed out")
        self.parts.append(name)
        return PartRef(run_id=self.run_id, name=name, parts_dir=str(self.parts_dir))

    def add_component(self, name: str, files, verify) -> None:
        valid_name(name, "component")
        if name in self.components:
            raise ValueError(f"component {name!r} is already registered")
        resolve(verify)  # fail now, in the script, not at the end of the run
        self.components[name] = (str(Path(files).resolve()), verify)

    def add_check(self, name: str, verify=None, ok=None, detail=None) -> None:
        if not name or any(c["name"] == name for c in self.checks):
            raise ValueError(f"check {name!r}: give each check its own name")
        if (verify is None) == (ok is None):
            raise ValueError(f"check {name!r}: pass verify= (run at verify()) or ok= (a result)")
        if verify is not None:
            resolve(verify)
            self.checks.append({"name": name, "target": verify})
        else:
            self.checks.append({"name": name, "ok": bool(ok), "detail": detail})

    # ------------------------------------------------------------------ events
    def _on_event(self, event) -> None:  # O(1): runs in RHAPSODY's dispatch loop
        self._seen[event.event_type] += 1
        if event.event_type == Barrier.__name__ and self._barrier_seen is not None:
            self._barrier_seen.set()
        elif event.event_type == "SessionEnded":
            # telemetry is stopping: was the engine already shutting down? (a stop by the
            # engine's shutdown, or by hand while the engine still ran)
            self._session_ended_seen = True
            self._engine_down_at_stop = bool(getattr(self.flow, "is_shutting_down", False))

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
    def record_end(self, ended_by: str, exception_type: str) -> None:
        """Queue RunEnded while telemetry still runs; called by MiniApp's exit, never awaits.

        Nothing is decided here: under a stop that is still writing out its queue the
        outcome is only known in verify().
        """
        self.ended_by, self.exception_type = ended_by, exception_type
        if self.started and getattr(self.manager, "_running", False):  # PRIVATE (drift-tested)
            self._emit(
                RunEnded,
                ended_by=ended_by,
                exception_type=exception_type,
                duration_s=time.time() - self._t0,
            )
            self._run_ended_queued = True

    def verify(self) -> MiniAppReport:
        """Verify the checkpoint after the script stopped telemetry; never raises."""
        try:
            self.report = self._verify()
            self._engine_reasons = self.report.reasons
            self.report = assemble(
                self.report,
                run_id=self.run_id,
                part=self.part_of.name if self.part_of else "",
                parameters=self.parameters,
                parts_dir=self.parts_dir,
                parts=self.parts,
                components=self.components,
                checks=self.checks,
            )
        except Exception as exc:  # last resort: verify() never raises
            self.report = MiniAppReport(
                session_id=str(getattr(self.flow, "uid", "")),
                checkpoint=None,
                telemetry_complete=False,
                ended_by=self.ended_by,
                reasons=(f"verification failed: {type(exc).__name__}: {exc}",),
            )
        self._write_report()
        self._write_part()
        self._write_manifest()
        self._log_report()
        return self.report

    def _verify(self) -> MiniAppReport:
        session = str(getattr(self.flow, "uid", ""))
        if self.attach_error is not None or not self.started:
            why = (
                f"not attached: {self.attach_error}"
                if self.attach_error is not None
                else "MiniApp was never entered"
            )
            return MiniAppReport(session, None, False, self.ended_by, reasons=(why,))
        if getattr(self.manager, "_running", False):  # PRIVATE (drift-tested)
            # the checkpoint is still buffered: counting it would only produce noise
            return MiniAppReport(
                self.manager.session_id,
                None,
                False,
                self.ended_by,
                reasons=(
                    "telemetry still running: call verify() after flow.shutdown() "
                    "and telemetry.stop()",
                ),
            )
        reasons, notes = list(self.reasons), list(self.notes)
        if not self._session_ended_seen:
            reasons.append(
                "telemetry did not finish writing its queue (stop() stalled or was cancelled)"
            )
        if not self._run_ended_queued:
            if self._engine_down_at_stop:
                notes.append(
                    "RunEnded not in the checkpoint: the engine shutdown stopped telemetry "
                    "before MiniApp's exit"
                )
            else:
                reasons.append(
                    "stopped_before_exit: telemetry was stopped while the engine was still "
                    "running"
                )
        try:
            return verify_checkpoint(
                self.checkpoint_dir,
                self.manager.session_id,
                emitted={k: v for k, v in self._emitted.items() if k.startswith(PREFIX)},
                seen=dict(self._seen),
                ended_by=self.ended_by,
                reasons=tuple(reasons),
                notes=tuple(notes),
            )
        except Exception as exc:
            reasons.append(f"verification failed: {type(exc).__name__}: {exc}")
            return MiniAppReport(
                self.manager.session_id,
                None,
                False,
                self.ended_by,
                reasons=tuple(reasons),
                notes=tuple(notes),
            )

    def _write_report(self) -> None:
        if self.checkpoint_dir is None:  # not attached: no checkpoint to write next to
            return
        try:
            out = self.checkpoint_dir / f"{self.report.session_id}.wfminiapi.report.json"
            out.write_text(json.dumps(self.report.to_dict(), indent=1, default=str) + "\n")
        except OSError as exc:
            log.warning("could not write the MiniApp report: %s", exc)

    def _write_part(self) -> None:
        """A part's report, into its run's parts directory, where the run's verify() reads it."""
        if self.part_of is None:
            return
        try:
            write_part(self.part_of, self.report.to_dict())
        except OSError as exc:
            log.warning("could not write part %s's report for its run: %s", self.part_of.name, exc)

    def _write_manifest(self) -> None:
        """What the run is made of, to verify it again offline; only for a run of more than
        one engine, or with components, checks or parameters."""
        if self.checkpoint_dir is None or not (
            self.parts or self.components or self.checks or self.parameters
        ):
            return
        r = self.report
        manifest = {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "session_id": r.session_id,
            "part": r.part,
            "checkpoint_dir": str(self.checkpoint_dir),
            "engine_reasons": list(self._engine_reasons),
            "parameters": self.parameters,
            "parts": list(self.parts),
            "parts_dir": str(self.parts_dir),
            "components": {
                name: {
                    "files": files,
                    "verify": r.components.get(name, {}).get("verify", ""),
                    "verdict": r.components.get(name, {}),
                }
                for name, (files, _) in self.components.items()
            },
            "checks": [dict(c) for c in r.checks],
        }
        try:
            write_manifest(self.checkpoint_dir / f"{r.session_id}{MANIFEST_SUFFIX}", manifest)
        except OSError as exc:
            log.warning("could not write the run's manifest: %s", exc)

    def _log_report(self) -> None:
        r, t = self.report, self.report.tasks
        msg = (
            f"MiniApp {r.session_id}: telemetry "
            f"{'complete' if r.telemetry_complete else 'INCOMPLETE'}; tasks created={t.created} "
            f"completed={t.completed} failed={t.failed} canceled={t.canceled} "
            f"abandoned={t.abandoned} not_submitted={t.not_submitted}"
        )
        if r.parts or r.components or r.checks:
            msg += (
                f"; parts={len(r.parts)}/{len(self.parts)} components={len(r.components)} "
                f"checks {'pass' if r.checks_pass else 'FAIL'}"
            )
        if r.reasons:
            msg += "; " + "; ".join(r.reasons)
        if r.notes:
            msg += "; note: " + "; ".join(r.notes)
        log.log(logging.INFO if r.telemetry_complete else logging.WARNING, msg)
