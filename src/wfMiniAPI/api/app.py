"""MiniApp: the single contact point of a workflow mini-app.

    backend   = await DragonExecutionBackend()
    flow      = await WorkflowEngine.create(backend=backend)
    telemetry = await flow.start_telemetry(checkpoint_path="telemetry")
    app = MiniApp(flow, telemetry)
    try:
        async with app, flow.workflow_scope("motif_4"):
            await miniapp_workflow(cfg, flow, logger)
    finally:
        try:
            await flow.shutdown()
        finally:
            await telemetry.stop()
            app.verify()

MiniApp is a thin observer of components the user built.  It never constructs, hides,
replaces or shuts them down: the script that created the engine and telemetry shuts
them down, and the nested ``finally`` stops telemetry and verifies even when the
engine's shutdown fails.  MiniApp records the run and verifies its telemetry
(``app.verify()``, ``app.report``).

A run of many engines and components: what the run is made of is declared on its
MiniApp, in ``app.composition`` (``wfMiniAPI.composition``), and its telemetry records and
verifies it. ``app.parameters`` (or ``MiniApp(..., parameters=...)``) records the
configuration. The run's MiniApp hands each other engine a ``PartRef`` (``app.part(name)``),
and that engine's own ``MiniApp(flow, telemetry, part_of=ref)`` reports into the run, from
any process. Components that are not engines register their files and a verifier
(``app.component``), and checks across them read the whole run (``app.check``).
"""

from __future__ import annotations

import asyncio
import atexit
import functools
import logging
import sys

from ..composition import Component, Composition, PartRef
from ..telemetry.capability import AttachError, MiniAppTelemetry
from ..telemetry.run import ComponentReport, Run
from ..telemetry.verify import MiniAppReport

__all__ = [
    "AttachError",
    "Component",
    "ComponentReport",
    "Composition",
    "MiniApp",
    "MiniAppReport",
    "PartRef",
    "Run",
]

log = logging.getLogger("wfMiniAPI.telemetry")


def _would_emit(logger: logging.Logger, level: int) -> bool:
    """True if a record at ``level`` reaches a handler that outputs it.

    With no handler at all, logging's last-resort handler prints it.  Only NullHandlers
    (the Dragon runtime adds one to the root logger) means nobody sees it.
    """
    found = False
    node: logging.Logger | None = logger
    while node is not None:
        for handler in node.handlers:
            found = True
            if not isinstance(handler, logging.NullHandler) and level >= handler.level:
                return True
        node = node.parent if node.propagate else None
    return not found


def _warn_unverified(session: str) -> None:
    """At interpreter exit, for a MiniApp whose verify() was never called."""
    if not log.isEnabledFor(logging.WARNING):
        return  # silenced on purpose
    msg = (
        f"MiniApp {session}: verify() was never called; no report written. "
        "Call app.verify() after flow.shutdown() and telemetry.stop()."
    )
    if _would_emit(log, logging.WARNING):
        log.warning(msg)
    else:  # only NullHandlers (e.g. under Dragon): the warning would vanish
        print(f"WARNING: {msg}", file=sys.stderr, flush=True)


class MiniApp:
    """Async context manager that records a workflow run and verifies its telemetry.

    On entry it checks that ``telemetry`` is the manager ``flow`` emits into and can be
    verified, and records the run.  On exit (normal, exception, cancel or signal) it
    only records how the run ended; it awaits nothing and shuts nothing down.  After
    the script has shut the engine down and stopped telemetry, ``verify()`` checks the
    checkpoint and sets ``report``.

    ``composition`` is what the run is made of beyond this engine's tasks: its
    configuration, the other engines it hands out as parts, its components and its
    checks. The methods below declare it; telemetry records and verifies it.
    """

    def __init__(
        self,
        flow,
        telemetry,
        /,
        *,
        backends=None,
        part_of: PartRef | None = None,
        parameters=None,
    ):
        self.flow = flow
        self.backends = list(backends) if backends is not None else [flow.backend]
        self.composition = Composition()
        self.telemetry = MiniAppTelemetry(
            flow, telemetry, composition=self.composition, part_of=part_of
        )
        if parameters is not None:
            self.parameters(parameters)
        # one stored object (atexit.unregister compares by equality) holding only a
        # string, so nothing of the run is kept alive until interpreter exit
        session = getattr(telemetry, "session_id", None) or getattr(flow, "uid", "")
        self._reminder = functools.partial(_warn_unverified, str(session))

    @property
    def report(self) -> MiniAppReport | None:
        """The verified telemetry report, set by ``verify()``."""
        return self.telemetry.report

    async def __aenter__(self) -> MiniApp:
        atexit.register(self._reminder)
        try:
            self.telemetry.check()
            await self.telemetry.start(self.backends)
        except BaseException as exc:
            self.telemetry.attach_error = f"{type(exc).__name__}: {exc}"
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        # No await: a cancel cannot interrupt the exit.  Never mask the workflow's outcome.
        try:
            self.telemetry.record_end(
                self._ended_by(exc_type), exc_type.__name__ if exc_type else ""
            )
        except Exception as err:
            self.telemetry.reasons.append(f"run-end event failed: {type(err).__name__}: {err}")
        return False

    # ------------------------------------------------------------------ a run of many parts
    def part(self, name: str) -> PartRef:
        """Hand another engine of this run a reference to report into it.

        Pass the reference to that engine's ``MiniApp(flow, telemetry, part_of=ref)``, in
        this process or another.  This run's ``verify()`` then requires that part's report:
        a part handed out that never reports makes the run's telemetry incomplete.  Call it
        inside ``async with app``.
        """
        if self.telemetry.part_of is not None:
            raise RuntimeError("a part cannot hand out parts: call part() on the run's MiniApp")
        parts_dir = self.telemetry.parts_dir
        if parts_dir is None:
            raise RuntimeError("enter the MiniApp (async with app) before handing out parts")
        self.composition.add_part(name)
        return PartRef(run_id=self.telemetry.run_id, name=name, parts_dir=str(parts_dir))

    def component(self, name: str, files, *, verify) -> None:
        """Register a component's files and its verifier.

        ``verify`` is ``"module:function"`` or a callable taking the files' path and
        returning a ``ComponentReport`` or a bool.  It runs in ``verify()``; named by
        import path, it also runs when the run is verified again offline.
        """
        self.composition.add_component(name, files, verify)

    def check(self, name: str, *, verify=None, ok: bool | None = None, detail=None) -> None:
        """Register a check on the run's results.

        Either ``verify``, ``"module:function"`` or a callable taking a ``Run`` and
        returning a bool or ``(bool, detail)``, run in ``verify()`` after the parts and
        components; or ``ok`` (and ``detail``), a result computed in the run.  Failed
        checks set ``report.checks_pass`` to False; they never make telemetry incomplete.
        """
        self.composition.add_check(name, verify, ok, detail)

    def parameters(self, values) -> None:
        """Record the run's configuration: in the composition, in the checkpoint (a
        ``wfminiapi.Parameters`` event, once telemetry runs) and in the report. Also
        ``MiniApp(..., parameters=values)``. A later call replaces the values."""
        self.composition.set_parameters(values)
        self.telemetry.record_parameters()

    def verify(self) -> MiniAppReport:
        """Verify the telemetry checkpoint and set ``report``; never raises.

        Call it after ``flow.shutdown()`` and ``telemetry.stop()``.  It also writes
        ``<checkpoint>/<session>.wfminiapi.report.json``.  Calling it again recomputes it.
        """
        atexit.unregister(self._reminder)
        return self.telemetry.verify()

    def _ended_by(self, exc_type) -> str:
        if exc_type is None:
            return "normal"
        if not issubclass(exc_type, asyncio.CancelledError):
            return "exception"
        if asyncio.current_task().cancelling():
            return "cancelled"  # task.cancel(): a timeout, TaskGroup, asyncio.run, supervisor
        if self.flow.is_shutting_down:
            return "engine_shutdown"  # AsyncFlow cancelled the futures: signal, error, call
        return "cancelled"
