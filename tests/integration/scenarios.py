"""MiniApp scenarios, one per process (run by test_miniapp.py).

    python scenarios.py <scenario> <workdir>                  # Concurrent backend
    WFMINI_SCENARIO_BACKEND=dragon dragon -s scenarios.py <scenario> <workdir>

Prints one JSON line prefixed with ``RESULT `` describing the outcome.
"""

import asyncio
import contextlib
import inspect
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend, DragonExecutionBackend
from rhapsody.telemetry.manager import TelemetryManager

from wfMiniAPI.api import AttachError, MiniApp

SCENARIO, WORKDIR = sys.argv[1], Path(sys.argv[2])
BACKEND = os.environ.get("WFMINI_SCENARIO_BACKEND", "concurrent")
result = {"scenario": SCENARIO, "backend": BACKEND}
OUTER_TIMEOUT = {"outer_cancel", "ddsim_outer_cancel"}  # asyncio.timeout(1) around the run


def done(**fields):
    result.update(fields)
    print("RESULT " + json.dumps(result, default=str), flush=True)


async def engine():
    if BACKEND == "dragon":
        backend = await DragonExecutionBackend()
    else:
        backend = await ConcurrentExecutionBackend()
    return await WorkflowEngine.create(backend=backend)


async def backend_state(flow) -> str:
    state = flow.backend.state()  # sync on some backends, async on others
    if inspect.isawaitable(state):
        state = await state
    return str(state).upper()


async def attach_error(flow, manager, *cleanup):
    """Enter MiniApp expecting AttachError, with the script's own nested finally; then shut
    down the other objects the scenario made, which that finally does not cover (on
    Dragon the process would not exit otherwise)."""
    app = MiniApp(flow, manager)
    try:
        async with app:
            pass
    except AttachError as exc:
        result["attach_error"] = str(exc)
    finally:
        try:
            await flow.shutdown()
        finally:
            await manager.stop()
            report = app.verify()
    for step in cleanup:
        await step()
    return done(attach_error=result.get("attach_error"), report=report.to_dict())


async def main():
    flow = await engine()
    telemetry = await flow.start_telemetry(checkpoint_path=str(WORKDIR / "telemetry"))

    @flow.function_task
    async def add(a, b):
        return a + b

    @flow.function_task
    async def boom():
        raise RuntimeError("task failed on purpose")

    @flow.executable_task
    async def false_exe():
        return "false"

    @flow.function_task
    async def wait_for_signal():
        (WORKDIR / "ready").write_text("1")
        time.sleep(3)

    @flow.function_task
    async def slow():
        time.sleep(2)

    @flow.block
    async def blk():
        return await add(1, 1)

    if SCENARIO == "late_attach":
        await add(1, 2)
        return await attach_error(flow, telemetry)

    if SCENARIO == "wrong_telemetry":
        other = await engine()
        return await attach_error(other, telemetry, flow.shutdown)

    if SCENARIO == "hand_built_no_start":
        other = await engine()  # never called start_telemetry()
        hand = TelemetryManager(session_id=other.uid, checkpoint_path=str(WORKDIR / "hand"))
        await hand.start()
        return await attach_error(other, hand, flow.shutdown, telemetry.stop)

    if SCENARIO == "hand_built_telemetry":
        hand = TelemetryManager(session_id=flow.uid, checkpoint_path=str(WORKDIR / "hand"))
        await hand.start()
        return await attach_error(flow, hand, telemetry.stop)

    if SCENARIO == "second_start_telemetry":
        second = await flow.start_telemetry(checkpoint_path=str(WORKDIR / "second"))
        return await attach_error(flow, telemetry, second.stop)

    if SCENARIO == "shutdown_fails":
        real_shutdown = flow.backend.shutdown

        async def failing_shutdown():  # shuts down, then reports a failure
            await real_shutdown()
            raise RuntimeError("backend shutdown failed")

        flow.backend.shutdown = failing_shutdown

    app = MiniApp(flow, telemetry)
    t0 = time.monotonic()
    main_task = asyncio.current_task()
    before_enter = random.getstate()
    before_exit = None
    outer = asyncio.timeout(1) if SCENARIO in OUTER_TIMEOUT else contextlib.nullcontext()
    try:
        # the script shape of design B: the script shuts down; MiniApp records and verifies
        try:
            async with outer:
                async with app, flow.workflow_scope("scenario"):
                    if SCENARIO == "random_state":
                        # MiniApp's own enter and exit, measured apart from the backend's
                        # start-up and shutdown (Dragon's runtime draws from `random` itself)
                        result["random_unchanged_by_enter"] = random.getstate() == before_enter
                        await flow.shutdown()
                        before_exit = random.getstate()
                    elif SCENARIO in ("ok", "no_verify", "shutdown_fails"):
                        await add(add(1, 2), 3)
                        await blk()
                    elif SCENARIO == "task_fail":
                        try:
                            await false_exe()
                        except Exception:
                            pass
                    elif SCENARIO == "dep_fail":
                        try:
                            await add(boom(), 1)
                        except Exception:
                            pass
                    elif SCENARIO == "raise":
                        await add(1, 1)
                        raise ValueError("core raised")
                    elif SCENARIO == "early_stop":
                        await add(1, 1)
                        await telemetry.stop()  # by hand, while the engine still runs
                    elif SCENARIO == "hand_stop_then_shutdown":
                        await add(1, 1)
                        await telemetry.stop()
                        await flow.shutdown()
                    elif SCENARIO == "engine_stopped_telemetry":
                        await add(1, 1)
                        await flow.shutdown()  # what AsyncFlow #125's shutdown() will do
                        await telemetry.stop()
                    elif SCENARIO == "core_shuts_down":  # a core that still shuts down
                        await add(1, 1)
                        await flow.shutdown()
                    elif SCENARIO == "verify_too_early":
                        await add(1, 1)
                        result["early_report"] = app.verify().to_dict()
                    elif SCENARIO in ("sigint", "sigterm"):
                        fut = wait_for_signal()
                        while not (WORKDIR / "ready").exists():
                            await asyncio.sleep(0.01)
                        sig = signal.SIGINT if SCENARIO == "sigint" else signal.SIGTERM
                        os.kill(os.getpid(), sig)
                        await fut
                    elif SCENARIO == "outer_cancel":
                        await slow()
                    elif SCENARIO == "ddsim_outer_cancel":
                        try:  # a core that shuts the engine down in its own finally
                            await slow()
                        finally:
                            await flow.shutdown()
                    elif SCENARIO == "kept_backend":
                        await add(1, 1)
                        await flow.shutdown(skip_execution_backend=True)  # kept for later
            if before_exit is not None:
                result["random_unchanged_by_exit"] = random.getstate() == before_exit
        finally:
            try:
                await flow.shutdown()
            finally:
                await telemetry.stop()
                if SCENARIO != "no_verify":
                    app.verify()
    except BaseException as exc:
        result["raised"] = type(exc).__name__
        raise
    finally:
        result["cancelling_after"] = main_task.cancelling()
        result["backend_state"] = await backend_state(flow)
        if SCENARIO == "kept_backend":  # the owner of the kept backend shuts it down
            await flow.backend.shutdown()
            result["backend_state_after_owner"] = await backend_state(flow)
        report = app.report.to_dict() if app.report else None
        done(report=report, elapsed=time.monotonic() - t0)


if BACKEND == "dragon":
    import multiprocessing as mp

    mp.set_start_method("dragon")

if SCENARIO == "random_state":
    # isolate MiniApp's own draws from OpenTelemetry's (UP-18): ids from os.urandom,
    # exemplars off -- the same neutralisation the equivalence shim uses
    os.environ["OTEL_METRICS_EXEMPLAR_FILTER"] = "always_off"
    from opentelemetry.sdk.trace import id_generator as _idg

    _idg.RandomIdGenerator.generate_span_id = lambda self: int.from_bytes(os.urandom(8), "big") or 1
    _idg.RandomIdGenerator.generate_trace_id = lambda self: (
        int.from_bytes(os.urandom(16), "big") or 1
    )
    _state = random.getstate()
    asyncio.run(main())
    print("RANDOM_UNCHANGED", random.getstate() == _state, flush=True)
else:
    asyncio.run(main())
