"""Verifying a run of many engines and components: its parts, its components, its checks.

What the run is made of is declared on its ``MiniApp`` (``wfMiniAPI.composition``): each
other engine of the run is a part, handed a ``PartRef`` (``app.part(name)``); that engine gets
a ``MiniApp`` of its own, ``MiniApp(flow, telemetry, part_of=ref)``, in whatever process it
runs, and its ``verify()`` writes its report into the run's parts directory. Components other
than engines (an agent library's OpenTelemetry export, a model server's log, the
application's results) register their files and a verifier (``app.component``). Checks across
all of them (``app.check``) read a ``Run``. This module only verifies and records them.

The run's ``verify()`` collects the parts' reports, runs each component's verifier over its
files, then runs the checks. Its report keeps two verdicts apart: ``telemetry_complete``
(every engine and component left complete records) and ``checks_pass`` (the results are
valid, the parts' own checks included). It also writes a manifest beside the report, from
which ``python -m wfMiniAPI.verify <dir>`` verifies the run again offline: verifiers and
checks named by import path run again; the others keep the verdict recorded in the run. The
manifest's paths are relative to its directory, so a run directory can be moved.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# What the run is made of; imported here too, where they have always been importable from.
from ..composition import Component, PartRef, jsonable, resolve, valid_name
from .verify import MiniAppReport, verify_checkpoint

__all__ = [
    "Component",
    "ComponentReport",
    "PartRef",
    "Run",
    "assemble",
    "jsonable",
    "resolve",
    "valid_name",
    "verify_run",
]

MANIFEST_SUFFIX = ".wfminiapi.manifest.json"


@dataclass(frozen=True)
class ComponentReport:
    """A component verifier's verdict on its files."""

    complete: bool
    reasons: tuple[str, ...] = ()  # why the component's records are incomplete
    notes: tuple[str, ...] = ()
    counts: dict = field(default_factory=dict)  # what the component recorded, for checks


@dataclass(frozen=True)
class Run:
    """What a check reads."""

    run_id: str
    parameters: dict  # the run's configuration
    parts: dict  # part name -> its engine's report (a dict, as in its report file)
    components: dict  # component name -> its verdict (complete, reasons, counts, files)
    report: dict  # the run's own engine's report


def _write_json(path: Path, data: Any) -> None:
    """Write whole or not at all: a reader on a shared file system never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(jsonable(data), indent=1) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------- parts
def write_part(ref: PartRef, report: dict) -> Path:
    """Write a part's report into its run's parts directory, one file per engine
    (``<name>.<session>.json``), so that two engines given the same part both show."""
    session = str(report.get("session_id") or "unknown")
    path = Path(ref.parts_dir) / f"{ref.name}.{session}.json"
    _write_json(path, {"run_id": ref.run_id, "part": ref.name, "report": report})
    return path


def collect_parts(parts_dir: str | Path, run_id: str, names) -> tuple[dict, list[str]]:
    """The reports of the parts handed out, and why the run is incomplete because of them."""
    parts: dict[str, dict] = {}
    reasons: list[str] = []
    for name in names:
        found = sorted(Path(parts_dir).glob(f"{name}.*.json"))
        if not found:
            reasons.append(f"part {name}: no report (it never ran, or never called verify())")
            continue
        if len(found) > 1:
            sessions = ", ".join(f.name[len(name) + 1 : -len(".json")] for f in found)
            reasons.append(f"part {name}: reported by {len(found)} engines ({sessions})")
            continue
        try:
            record = json.loads(found[0].read_text())
            if not isinstance(record, dict) or not isinstance(record.get("report"), dict):
                raise ValueError("not a part's report")
        except (OSError, ValueError) as exc:
            reasons.append(f"part {name}: unreadable report: {exc}")
            continue
        if record.get("run_id") != run_id:
            reasons.append(f"part {name}: its report belongs to run {record.get('run_id')!r}")
            continue
        report = record["report"]
        parts[name] = report
        if not report.get("telemetry_complete"):
            why = "; ".join(report.get("reasons") or ()) or "telemetry incomplete"
            reasons.append(f"part {name}: {why}")
    return parts, reasons


def part_check_failures(parts: dict) -> list[dict]:
    """One failed check per part whose own checks failed: the run is valid only if its
    parts are."""
    failed = []
    for name, report in parts.items():
        if report.get("checks_pass") is False:
            names = [c.get("name") for c in report.get("checks") or () if not c.get("ok")]
            failed.append({"name": f"part {name}", "ok": False, "detail": names, "verify": ""})
    return failed


# ---------------------------------------------------------------------------- components
def verify_components(components: dict[str, Component]) -> tuple[dict, list[str]]:
    """Run each component's verifier over its files: ``{name: Component}``.

    A verifier takes the files' path and returns a ``ComponentReport`` or a bool. One that
    raises makes its component incomplete; it never stops the others.
    """
    verdicts: dict[str, dict] = {}
    reasons: list[str] = []
    for name, component in components.items():
        files, target = component.files, component.verify
        path = target if isinstance(target, str) else ""
        try:
            path, fn = resolve(target)
            verdict = fn(Path(files))
            if isinstance(verdict, bool):
                verdict = ComponentReport(verdict)
            if not isinstance(verdict, ComponentReport):
                raise TypeError(
                    f"the verifier returned {type(verdict).__name__}, not ComponentReport or bool"
                )
        except Exception as exc:  # noqa: BLE001 - a verifier's failure is a verdict
            verdict = ComponentReport(False, reasons=(f"{type(exc).__name__}: {exc}",))
        verdicts[name] = {"files": str(files), "verify": path, **jsonable(verdict)}
        if not verdict.complete:
            why = verdicts[name]["reasons"] or ["incomplete"]
            reasons += [f"component {name}: {r}" for r in why]
    return verdicts, reasons


def _recorded(verdict: dict, files) -> ComponentReport:
    """A verdict from the run, for a verifier that cannot run again offline."""
    return ComponentReport(
        bool(verdict.get("complete")),
        reasons=tuple(verdict.get("reasons") or ()),
        notes=(*(verdict.get("notes") or ()), "verified in the run, not again offline"),
        counts=dict(verdict.get("counts") or {}),
    )


# ---------------------------------------------------------------------------- checks
def run_checks(checks: list[dict], run: Run) -> list[dict]:
    """Run each check: ``{"name", "target"}`` (a callable or ``"module:function"`` that takes
    a ``Run`` and returns a bool or ``(bool, detail)``), or a result recorded as it is
    (``{"name", "ok", "detail"}``). A check that raises fails."""
    results = []
    for check in checks:
        name = check["name"]
        target = check.get("target") or check.get("verify")
        if not target:
            ok, detail, path = bool(check.get("ok")), check.get("detail"), ""
        else:
            path = target if isinstance(target, str) else ""
            try:
                path, fn = resolve(target)
                outcome = fn(run)
                ok, detail = (
                    outcome if isinstance(outcome, tuple) and len(outcome) == 2 else (outcome, None)
                )
                ok = bool(ok)
            except Exception as exc:  # noqa: BLE001 - a check that raises fails
                ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append({"name": name, "ok": ok, "detail": jsonable(detail), "verify": path})
    return results


# ---------------------------------------------------------------------------- the run
def assemble(
    base: MiniAppReport,
    *,
    run_id: str,
    part: str = "",
    parameters: dict | None = None,
    parts_dir: str | Path | None = None,
    parts=(),
    components: dict[str, Component] | None = None,
    checks: list[dict] | None = None,
) -> MiniAppReport:
    """The run's report: its engine's report, with its parts, components and checks."""
    reasons, notes = list(base.reasons), list(base.notes)
    found: dict = {}
    if parts:
        found, why = collect_parts(parts_dir, run_id, parts)
        reasons += why
    verdicts, why = verify_components(components or {})
    reasons += why
    parameters = jsonable(dict(parameters or {}))
    run = Run(run_id, parameters, found, verdicts, base.to_dict())
    results = run_checks(list(checks or ()), run) + part_check_failures(found)
    return dataclasses.replace(
        base,
        telemetry_complete=base.telemetry_complete and len(reasons) == len(base.reasons),
        reasons=tuple(reasons),
        notes=tuple(notes),
        run_id=run_id,
        part=part,
        parameters=parameters,
        parts=found,
        components=verdicts,
        checks=tuple(results),
        checks_pass=all(r["ok"] for r in results),
    )


def write_manifest(path: str | Path, manifest: dict) -> None:
    _write_json(Path(path), manifest)


def _no_run(why: str) -> MiniAppReport:
    return MiniAppReport("", None, False, reasons=(why,))


def _pick_manifest(d: Path, session: str | None) -> tuple[dict | None, str]:
    """The manifest to verify: the session's, or the only run's (parts' manifests aside)."""
    manifests = []
    for path in sorted(d.glob(f"*{MANIFEST_SUFFIX}")):
        try:
            m = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(m, dict) and m.get("session_id"):
            manifests.append(m)
    if session:
        for m in manifests:
            if m["session_id"] == session:
                return m, ""
        return None, f"no manifest for session {session} in {d}"
    runs = [m for m in manifests if not m.get("part")] or manifests
    if len(runs) == 1:
        return runs[0], ""
    sessions = ", ".join(m["session_id"] for m in runs) or "none"
    return None, (
        f"expected one run's *{MANIFEST_SUFFIX} in {d}, found {len(runs)} "
        f"(sessions: {sessions}): pass the session to verify"
    )


def _where(d: Path, relative: str | None, absolute: str | None) -> Path:
    """A path the manifest recorded: relative to its directory, else where it was."""
    here = d / (relative or "")
    if relative and (here.exists() or not absolute):
        return here
    return Path(absolute) if absolute else here


def verify_run(directory: str | Path, session: str | None = None) -> MiniAppReport:
    """Verify a run again from its files: the manifest in ``directory`` (the run engine's
    checkpoint directory), its checkpoint, its parts' reports, its components' files.
    ``session`` picks the run when the directory holds more than one."""
    d = Path(directory)
    m, why = _pick_manifest(d, session)
    if m is None:
        return _no_run(why)
    base = verify_checkpoint(
        d,
        m["session_id"],
        reasons=tuple(m.get("engine_reasons") or ()),
        notes=("verified offline: the checkpoint is not compared with what was emitted",),
    )
    components = {}
    for name, c in (m.get("components") or {}).items():
        target = c.get("verify") or functools.partial(_recorded, c.get("verdict") or {})
        files = _where(d, c.get("files"), c.get("files_abs"))
        components[name] = Component(name, str(files), target)
    return assemble(
        base,
        run_id=m["run_id"],
        part=m.get("part", ""),
        parameters=m.get("parameters"),
        parts_dir=_where(d, m.get("parts_dir"), m.get("parts_dir_abs")),
        parts=m.get("parts") or (),
        components=components,
        checks=m.get("checks"),
    )
