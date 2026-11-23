"""What a run is made of, declared on its ``MiniApp``.

A workflow mini-app's run can be more than one engine's tasks. It can also hold other engines
(a population of agents, each population with its engine, in its own process) and components
that are not engines but leave records (an agent library's telemetry export, a model server's
counters, the application's results). The run's ``Composition`` names them:

- **parameters:** the run's configuration;
- **parts:** the other engines of the run, each handed a ``PartRef``;
- **components:** what is not an engine, as a ``Component`` (its records and their verifier);
- **checks:** functions over the whole run, its results included.

``MiniApp`` holds the composition (``app.composition``) and declares it with
``app.parameters``, ``app.part``, ``app.component`` and ``app.check``. Its telemetry records it
and verifies it (``wfMiniAPI.telemetry.run``). This module imports neither, and no framework.
"""

from __future__ import annotations

import dataclasses
import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class PartRef:
    """What another engine of the run needs to report into it. Picklable: pass it to a
    process or a task that builds that engine. Give each engine its own part."""

    run_id: str  # the run's engine session id
    name: str  # the part's name, unique in the run
    parts_dir: str  # where the part's report goes: an absolute path


@dataclass(frozen=True)
class Component:
    """A component of the run that is not an engine: where its records are, and the verifier
    that judges them."""

    name: str
    files: str  # the component's records, a file or a directory: an absolute path
    # "module:function", or a callable: takes the files' path, returns a ComponentReport or
    # a bool. Named by import path, it also runs when the run is verified again offline.
    verify: str | Callable


@dataclass
class Composition:
    """What a run is made of: its configuration, its parts, its components, its checks.

    Its ``MiniApp`` fills it, refusing at the call what the run could not verify. A run of
    one engine with none of these declares nothing (``declared`` is False) and sees no change.
    """

    parameters: dict = field(default_factory=dict)
    parts: list[str] = field(default_factory=list)
    components: dict[str, Component] = field(default_factory=dict)
    # {"name", "target"} (run at verify()) or {"name", "ok", "detail"} (a result)
    checks: list[dict] = field(default_factory=list)

    @property
    def declared(self) -> bool:
        """True once the run declares more than its one engine's tasks."""
        return bool(self.parameters or self.parts or self.components or self.checks)

    def set_parameters(self, values) -> None:
        self.parameters = jsonable(dict(values))

    def add_part(self, name: str) -> None:
        valid_name(name, "part")
        if name in self.parts:
            raise ValueError(f"part {name!r} was already handed out")
        self.parts.append(name)

    def add_component(self, name: str, files, verify) -> Component:
        valid_name(name, "component")
        if name in self.components:
            raise ValueError(f"component {name!r} is already registered")
        resolve(verify)  # fail now, in the script, not at the end of the run
        component = Component(name, str(Path(files).resolve()), verify)
        self.components[name] = component
        return component

    def add_check(self, name: str, verify=None, ok=None, detail=None) -> None:
        if not name or any(c["name"] == name for c in self.checks):
            raise ValueError(f"check {name!r}: give each check its own name")
        if name.startswith("part "):
            raise ValueError(f"check {name!r}: names starting 'part ' are the parts' checks")
        if (verify is None) == (ok is None):
            raise ValueError(f"check {name!r}: pass verify= (run at verify()) or ok= (a result)")
        if verify is not None:
            resolve(verify)
            self.checks.append({"name": name, "target": verify})
        else:
            self.checks.append({"name": name, "ok": bool(ok), "detail": jsonable(detail)})


def valid_name(name: str, what: str) -> str:
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"{what} name {name!r}: use letters, digits, '_' or '-'")
    return name


def jsonable(obj: Any) -> Any:
    """``obj`` as plain JSON data: keys become strings, sets and tuples lists, and anything
    else its ``str()``. Details and counts come from user code; reports must always write."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k if isinstance(k, str) else str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return [jsonable(v) for v in sorted(obj, key=repr)]
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    return str(obj)


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
