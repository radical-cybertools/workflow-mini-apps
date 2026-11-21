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
"""

from __future__ import annotations

import asyncio
import atexit
import functools
import logging
import sys

from ..telemetry.capability import AttachError, MiniAppTelemetry
from ..telemetry.verify import MiniAppReport

__all__ = ["AttachError", "MiniApp", "MiniAppReport"]

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
    """

    def __init__(self, flow, telemetry, /, *, backends=None):
        self.flow = flow
        self.backends = list(backends) if backends is not None else [flow.backend]
        self.telemetry = MiniAppTelemetry(flow, telemetry)
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
