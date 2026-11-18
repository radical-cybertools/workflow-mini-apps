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
import inspect

from ..telemetry.capability import AttachError, MiniAppTelemetry
from ..telemetry.verify import MiniAppReport

__all__ = ["AttachError", "MiniApp", "MiniAppReport"]

_POLL_S = 0.05


async def _backend_state(backend) -> str:
    state = backend.state()  # BaseBackend declares it sync; Concurrent/Dragon make it async
    if inspect.isawaitable(state):
        state = await state
    return str(state)


class MiniApp:
    """Async context manager tying a workflow run to its telemetry.

    On entry it checks that ``telemetry`` belongs to ``flow`` and can be verified, and
    records the run.  On exit (normal, exception or signal) it makes sure the engine is
    shut down, records the end of the run, stops telemetry and verifies the checkpoint.
    It never raises on exit and never replaces the workflow's own exception.
    """

    def __init__(self, flow, telemetry, /, *, backends=None, shutdown_wait_s: float = 60.0):
        self.flow = flow
        self.backends = list(backends) if backends is not None else [flow.backend]
        self.shutdown_wait_s = shutdown_wait_s
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
        interrupted = exc_type is not None and issubclass(exc_type, asyncio.CancelledError)
        reasons: list[str] = []
        try:
            await self._ensure_shutdown(exc_type, interrupted, reasons)
        except BaseException as err:  # the workflow's outcome must win
            reasons.append(f"engine shutdown check failed: {type(err).__name__}: {err}")
        await self.telemetry.finish(
            interrupted=interrupted,
            exception_type=exc_type.__name__ if exc_type else "",
            reasons=reasons,
        )
        return False

    async def _ensure_shutdown(self, exc_type, interrupted: bool, reasons: list[str]) -> None:
        if await self._is_shut_down():
            await self._wait_components_cleared(reasons)
            return
        if interrupted:
            # Signal path: AsyncFlow's own handler is already shutting the engine down.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.shutdown_wait_s
            while loop.time() < deadline and not await self._is_shut_down():
                await asyncio.sleep(_POLL_S)
            if await self._is_shut_down():
                await self._wait_components_cleared(reasons)
                return
            reasons.append("engine did not shut down after the signal; shutting it down")
        await self.flow.shutdown()
        await self._wait_components_cleared(reasons)

    async def _is_shut_down(self) -> bool:
        for backend in self.backends:
            if (await _backend_state(backend)).upper() != "SHUTDOWN":
                return False
        return True

    async def _wait_components_cleared(self, reasons: list[str]) -> None:
        # SHUTDOWN is set at the start of teardown; AsyncFlow clears its
        # components when its shutdown has finished.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.shutdown_wait_s
        while self.flow.components and loop.time() < deadline:
            await asyncio.sleep(_POLL_S)
        if self.flow.components:
            reasons.append("engine shutdown did not finish within shutdown_wait_s")
