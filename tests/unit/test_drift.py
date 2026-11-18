"""Pin the framework behaviour MiniApp relies on (AsyncFlow and RHAPSODY main).

If one of these fails after a framework upgrade, update MiniApp's touchpoints
(src/wfMiniAPI/telemetry/capability.py) before relaxing the pins.
"""

import asyncio
import inspect

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend, DragonExecutionBackend
from rhapsody.telemetry.manager import TelemetryManager


def test_telemetry_manager_private_touchpoints(tmp_path):
    async def go():
        tm = TelemetryManager(session_id="s.drift", checkpoint_path=str(tmp_path))
        assert tm._checkpoint_path == str(tmp_path)  # read by MiniApp until U-H lands
        await tm.start()
        assert tm._running is True
        files = list(tmp_path.glob("s.drift.*.telemetry.jsonl"))  # checkpoint file name
        await tm.stop()
        assert tm._running is False
        return files

    assert len(asyncio.run(go())) == 1
    assert "drain_on_stop" in inspect.signature(TelemetryManager.__init__).parameters


def test_asyncflow_public_surface_used_by_miniapp():
    async def go():
        flow = await WorkflowEngine.create(backend=await ConcurrentExecutionBackend())
        try:
            for attr in ("running", "resolved", "components", "backend", "uid"):
                assert hasattr(flow, attr), attr

            @flow.function_task
            async def t():
                return 1

            @flow.block
            async def b():
                return 1

            fut = t()
            assert getattr(fut, "uid", "").startswith("task.")  # uid at call time
            await fut
            await b()
            assert any(u.startswith("block.") for u in flow.components)  # block uid marker
        finally:
            await flow.shutdown()

    asyncio.run(go())


def test_asyncflow_joinable_shutdown_used_by_miniapp(tmp_path):
    """MiniApp's exit is one `await flow.shutdown()`, joining any shutdown running."""

    class Counting(ConcurrentExecutionBackend):
        calls = 0

        async def shutdown(self):
            type(self).calls += 1
            await asyncio.sleep(0.2)
            await super().shutdown()

    async def go():
        flow = await WorkflowEngine.create(backend=await Counting())
        assert flow.telemetry is None and not flow.is_shutting_down
        tm = await flow.start_telemetry(checkpoint_path=str(tmp_path))
        assert flow.telemetry is tm  # the manager the engine emits into
        waiter = asyncio.create_task(flow.shutdown())
        await asyncio.sleep(0)
        assert flow.is_shutting_down
        waiter.cancel()  # a cancelled waiter does not stop the shutdown
        await flow.shutdown()  # joins it
        assert Counting.calls == 1 and waiter.cancelled()
        assert str(await flow.backend.state()).upper() == "SHUTDOWN"
        await tm.stop()

    asyncio.run(go())
    assert hasattr(asyncio.Task, "get_context")  # the exit runs in the caller's Context


def test_dragon_backend_name_always_exported():
    # The equivalence shim must replace it, not add it (it exists even without Dragon).
    assert DragonExecutionBackend is not None
