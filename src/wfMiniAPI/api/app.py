"""MiniApp: the single contact point of a workflow mini-app.

    backend   = await DragonExecutionBackend()
    flow      = await WorkflowEngine.create(backend=backend)
    telemetry = await flow.start_telemetry(checkpoint_path="telemetry")
    async with MiniApp(flow, telemetry), flow.workflow_scope("motif_4"):
        await miniapp_workflow(cfg, flow, logger)

MiniApp is a thin coordinator over components the user built.  It never constructs,
hides or replaces them; it only orders the entry and exit of its capabilities
(today: telemetry, ``app.telemetry`` and ``app.report``).
"""

from __future__ import annotations

import asyncio

from ..telemetry.capability import AttachError, MiniAppTelemetry
from ..telemetry.verify import MiniAppReport

__all__ = ["AttachError", "MiniApp", "MiniAppReport"]


class MiniApp:
    """Async context manager tying a workflow run to its telemetry.

    On entry it checks that ``telemetry`` is the manager ``flow`` emits into and can be
    verified, and records the run.  On exit (normal, exception, cancel or signal) it
    shuts the engine down (``flow.shutdown()``, which joins a shutdown already running),
    records the end of the run, stops telemetry and verifies the checkpoint, in that
    order.  A cancel that arrives during the exit is delivered once the exit is done;
    the workflow's own exception wins over it.
    """

    def __init__(self, flow, telemetry, /, *, backends=None):
        self.flow = flow
        self.backends = list(backends) if backends is not None else [flow.backend]
        self.telemetry = MiniAppTelemetry(flow, telemetry)

    @property
    def report(self) -> MiniAppReport | None:
        """The verified telemetry report, available after the ``async with`` block."""
        return self.telemetry.report

    async def __aenter__(self) -> MiniApp:
        self.telemetry.check()
        await self.telemetry.start(self.backends)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        ended_by = self._ended_by(exc_type)  # before any await
        task = asyncio.current_task()
        # The caller's Context: RHAPSODY's telemetry.stop() detaches what start() attached
        exit_task = asyncio.get_running_loop().create_task(
            self._exit(exc_type, ended_by), context=task.get_context()
        )
        absorbed = 0  # cancels that arrived during the exit
        while not exit_task.done():
            try:
                await asyncio.shield(exit_task)
            except asyncio.CancelledError:
                if exit_task.cancelled():
                    break
                absorbed += 1  # finish the ordered stop first
        if absorbed and (exc_type is None or issubclass(exc_type, asyncio.CancelledError)):
            raise asyncio.CancelledError  # deliver the cancel, after the ordered stop
        for _ in range(absorbed):  # the workflow's exception wins: give the cancels back
            task.uncancel()
        return False

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

    async def _exit(self, exc_type, ended_by: str) -> None:
        reasons: list[str] = []
        try:
            await self.flow.shutdown()  # joins a shutdown already running
        except (Exception, asyncio.CancelledError) as err:  # the workflow's outcome wins
            reasons.append(f"engine shutdown failed: {type(err).__name__}: {err}")
        await self.telemetry.finish(
            ended_by=ended_by,
            exception_type=exc_type.__name__ if exc_type else "",
            reasons=reasons,
        )
