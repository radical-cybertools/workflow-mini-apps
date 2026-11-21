"""A run of many engines and components: its parts, its components' verifiers, its checks.

A run's ``MiniApp`` can hand each other engine of the run a ``PartRef`` (``app.part(name)``).
That engine gets a ``MiniApp`` of its own, ``MiniApp(flow, telemetry, part_of=ref)``, in
whatever process it runs, and its ``verify()`` writes its report into the run's parts
directory. Components other than engines (an agent library's OpenTelemetry export, a model
server's log, the application's results) register their files and a verifier
(``app.component``). Checks across all of them (``app.check``) read a ``Run``.

The run's ``verify()`` collects the parts' reports, runs each component's verifier over its
files, then runs the checks. Its report keeps two verdicts apart: ``telemetry_complete``
(every engine and component left complete records) and ``checks_pass`` (the results are
valid). It also writes a manifest beside the report, from which ``python -m wfMiniAPI.verify
<dir>`` verifies the run again offline: verifiers and checks named by import path run again;
the others keep the verdict recorded in the run.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .verify import MiniAppReport, verify_checkpoint

MANIFEST_SUFFIX = ".wfminiapi.manifest.json"
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class PartRef:
    """What another engine of the run needs to report into it. Picklable: pass it to a
    process or a task that builds that engine."""

    run_id: str  # the run's engine session id
    name: str  # the part's name, unique in the run
    parts_dir: str  # where the part's report goes


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


def valid_name(name: str, what: str) -> str:
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"{what} name {name!r}: use letters, digits, '_', '-' or '.'")
    return name


def resolve(target: str | Callable) -> tuple[str, Callable]:
    """``(import path, callable)`` for ``"module:function"`` or a callable.

    The path is ``""`` for a callable that cannot be imported again by name (a lambda, a
    local function, one defined in ``__main__``): it runs in the run, not offline.
    """
    if isinstance(target, str):
        module, sep, name = target.partition(":")
        if not (sep and module and name):
            raise ValueError(f"expected 'module:function', got {target!r}")
        obj: Any = importlib.import_module(module)
        for attr in name.split("."):
            obj = getattr(obj, attr)
        if not callable(obj):
            raise TypeError(f"{target} is not callable")
        return target, obj
    if not callable(target):
        raise TypeError(f"expected 'module:function' or a callable, got {target!r}")
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", "")
    if not module or module == "__main__" or "<" in qualname:
        return "", target
    path = f"{module}:{qualname}"
    try:
        _, found = resolve(path)
    except Exception:  # noqa: BLE001 - not importable by name: run it, but not offline
        return "", target
    return (path if found is target else ""), target


def _write_json(path: Path, data: Any) -> None:
    """Write whole or not at all: a reader on a shared file system never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------- parts
def write_part(ref: PartRef, report: dict) -> Path:
    """Write a part's report into its run's parts directory."""
    path = Path(ref.parts_dir) / f"{ref.name}.json"
    _write_json(path, {"run_id": ref.run_id, "part": ref.name, "report": report})
    return path


def collect_parts(parts_dir: str | Path, run_id: str, names) -> tuple[dict, list[str]]:
    """The reports of the parts handed out, and why the run is incomplete because of them."""
    parts: dict[str, dict] = {}
    reasons: list[str] = []
    for name in names:
        path = Path(parts_dir) / f"{name}.json"
        try:
            record = json.loads(path.read_text())
        except FileNotFoundError:
            reasons.append(f"part {name}: no report (it never ran, or never called verify())")
            continue
        except (OSError, ValueError) as exc:
            reasons.append(f"part {name}: unreadable report: {exc}")
            continue
        if record.get("run_id") != run_id:
            reasons.append(f"part {name}: its report belongs to run {record.get('run_id')!r}")
            continue
        report = record.get("report") or {}
        parts[name] = report
        if not report.get("telemetry_complete"):
            why = "; ".join(report.get("reasons") or ()) or "telemetry incomplete"
            reasons.append(f"part {name}: {why}")
    return parts, reasons


# ---------------------------------------------------------------------------- components
def verify_components(components: dict) -> tuple[dict, list[str]]:
    """Run each component's verifier over its files: ``{name: (files, target)}``.

    A verifier takes the files' path and returns a ``ComponentReport`` or a bool. One that
    raises makes its component incomplete; it never stops the others.
    """
    verdicts: dict[str, dict] = {}
    reasons: list[str] = []
    for name, (files, target) in components.items():
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
        verdicts[name] = {"files": str(files), "verify": path, **dataclasses.asdict(verdict)}
        if not verdict.complete:
            reasons += [f"component {name}: {r}" for r in verdict.reasons or ("incomplete",)]
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
            results.append(
                {
                    "name": name,
                    "ok": bool(check.get("ok")),
                    "detail": check.get("detail"),
                    "verify": "",
                }
            )
            continue
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
        results.append({"name": name, "ok": ok, "detail": detail, "verify": path})
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
    components: dict | None = None,
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
    parameters = dict(parameters or {})
    run = Run(run_id, parameters, found, verdicts, base.to_dict())
    results = run_checks(list(checks or ()), run)
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


def verify_run(directory: str | Path) -> MiniAppReport:
    """Verify a run again from its files: the manifest in ``directory`` (the run engine's
    checkpoint directory), its checkpoint, its parts' reports, its components' files."""
    d = Path(directory)
    found = sorted(d.glob(f"*{MANIFEST_SUFFIX}"))
    if len(found) != 1:
        return MiniAppReport(
            "",
            None,
            False,
            reasons=(f"expected one *{MANIFEST_SUFFIX} in {d}, found {len(found)}",),
        )
    m = json.loads(found[0].read_text())
    base = verify_checkpoint(
        m.get("checkpoint_dir") or d,
        m["session_id"],
        reasons=tuple(m.get("engine_reasons") or ()),
        notes=("verified offline: the checkpoint is not compared with what was emitted",),
    )
    components = {}
    for name, c in (m.get("components") or {}).items():
        target = c.get("verify") or functools.partial(_recorded, c.get("verdict") or {})
        components[name] = (c["files"], target)
    return assemble(
        base,
        run_id=m["run_id"],
        part=m.get("part", ""),
        parameters=m.get("parameters"),
        parts_dir=m.get("parts_dir"),
        parts=m.get("parts") or (),
        components=components,
        checks=m.get("checks"),
    )
