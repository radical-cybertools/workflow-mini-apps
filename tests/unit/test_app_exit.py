"""MiniApp's exit logic against a fake engine (no backend, no telemetry)."""

import asyncio

import pytest

from wfMiniAPI.api import AttachError, MiniApp


class FakeFlow:
    """What MiniApp's exit uses of an AsyncFlow engine; shutdown() takes `delay` s."""

    uid = "fake"
    backend = None
    telemetry = None

    def __init__(self, shutting_down=False, delay=0.0):
        self.is_shutting_down = shutting_down
        self.delay = delay
        self.shutdowns = 0

    async def shutdown(self):
        self.shutdowns += 1
        await asyncio.sleep(self.delay)
        self.is_shutting_down = True


class FakeTelemetry:
    """Stands in for MiniAppTelemetry: records what finish() was told."""

    report = None

    def __init__(self, flow):
        self.flow = flow
        self.finished = None

    async def finish(self, **kwargs):
        assert self.flow.shutdowns, "telemetry must stop after the engine shut down"
        self.finished = kwargs


def app_for(flow):
    app = MiniApp(flow, object())
    app.telemetry = FakeTelemetry(flow)
    return app


@pytest.mark.parametrize(
    "exc_type,shutting_down,expected",
    [
        (None, False, "normal"),
        (None, True, "normal"),  # the core shut the engine down itself (or caught a signal)
        (ValueError, False, "exception"),
        (KeyboardInterrupt, True, "exception"),
        (ExceptionGroup, False, "exception"),
        (asyncio.CancelledError, True, "engine_shutdown"),  # AsyncFlow cancelled the futures
        (asyncio.CancelledError, False, "cancelled"),
    ],
)
def test_ended_by(exc_type, shutting_down, expected):
    async def go():
        return app_for(FakeFlow(shutting_down))._ended_by(exc_type)

    assert asyncio.run(go()) == expected


def test_outer_cancel_after_the_cores_own_shutdown_is_cancelled():
    # DeepDriveSim's core awaits flow.shutdown() in its finally: the engine is shutting
    # down, but the run was cancelled from outside (a timeout calls task.cancel())
    async def go():
        app = app_for(FakeFlow(shutting_down=True))

        async def body():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                return app._ended_by(asyncio.CancelledError)

        task = asyncio.create_task(body())
        await asyncio.sleep(0)
        task.cancel()
        return await task

    assert asyncio.run(go()) == "cancelled"


def cancel_during_exit(exc_type):
    """Run __aexit__ with a slow engine shutdown and cancel it midway."""

    async def go():
        flow = FakeFlow(delay=0.2)
        app = app_for(flow)

        async def leave():
            exc = exc_type("core raised") if exc_type else None
            returned = await app.__aexit__(exc_type, exc, None)
            return returned, asyncio.current_task().cancelling()

        task = asyncio.create_task(leave())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            outcome = await task
        except asyncio.CancelledError:
            outcome = "CancelledError"
        return outcome, flow, app

    return asyncio.run(go())


def test_a_cancel_during_a_normal_exit_is_delivered_after_the_stop():
    outcome, flow, app = cancel_during_exit(None)
    assert outcome == "CancelledError"
    assert flow.shutdowns == 1 and app.telemetry.finished["ended_by"] == "normal"


def test_the_workflow_exception_wins_over_a_cancel_during_exit():
    outcome, flow, app = cancel_during_exit(ValueError)
    assert outcome == (False, 0)  # the ValueError propagates; the cancel was given back
    assert app.telemetry.finished["ended_by"] == "exception"


def test_a_failing_engine_shutdown_is_a_reason_not_an_exception():
    class Failing(FakeFlow):
        async def shutdown(self):
            self.shutdowns += 1
            raise RuntimeError("backend gone")

    async def go():
        app = app_for(Failing())
        assert await app.__aexit__(None, None, None) is False
        return app.telemetry.finished["reasons"]

    assert asyncio.run(go()) == ["engine shutdown failed: RuntimeError: backend gone"]


def test_an_asyncflow_without_joinable_shutdown_is_refused():
    class OldFlow:
        uid = "old"
        backend = None

    with pytest.raises(AttachError, match="no joinable shutdown"):
        MiniApp(OldFlow(), object()).telemetry.check()
