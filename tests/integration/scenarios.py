"""MiniApp scenarios, one per process (run by test_miniapp.py).

    python scenarios.py <scenario> <workdir>                  # Concurrent backend
    WFMINI_SCENARIO_BACKEND=dragon dragon -s scenarios.py <scenario> <workdir>

Prints one JSON line prefixed with ``RESULT `` describing the outcome.
"""

import asyncio
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend, DragonExecutionBackend

from wfMiniAPI.api import AttachError, MiniApp

SCENARIO, WORKDIR = sys.argv[1], Path(sys.argv[2])
BACKEND = os.environ.get("WFMINI_SCENARIO_BACKEND", "concurrent")
result = {"scenario": SCENARIO, "backend": BACKEND}


def done(**fields):
    result.update(fields)
    print("RESULT " + json.dumps(result, default=str), flush=True)


async def engine():
    if BACKEND == "dragon":
        backend = await DragonExecutionBackend()
    else:
        backend = await ConcurrentExecutionBackend()
    return await WorkflowEngine.create(backend=backend)


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

    @flow.block
    async def blk():
        return await add(1, 1)

    if SCENARIO == "late_attach":
        await add(1, 2)
        try:
            async with MiniApp(flow, telemetry):
                pass
        except AttachError as exc:
            result["attach_error"] = str(exc)
        # MiniApp never took over the engine or telemetry: shutting them down stays the
        # script's job (on Dragon the process would not exit otherwise)
        await flow.shutdown()
        await telemetry.stop()
        return done(attach_error=result.get("attach_error"))

    if SCENARIO == "wrong_telemetry":
        other = await engine()
        try:
            async with MiniApp(other, telemetry):
                pass
        except AttachError as exc:
            result["attach_error"] = str(exc)
        await other.shutdown()
        await flow.shutdown()
        await telemetry.stop()
        return done(attach_error=result.get("attach_error"))

    app = MiniApp(flow, telemetry)
    t0 = time.monotonic()
    before_enter = random.getstate()
    before_exit = None
    try:
        async with app, flow.workflow_scope("scenario"):
            if SCENARIO == "random_state":
                # MiniApp's own enter and exit, measured apart from the backend's start-up
                # and shutdown (Dragon's runtime draws from `random` itself)
                result["random_unchanged_by_enter"] = random.getstate() == before_enter
                await flow.shutdown()
                before_exit = random.getstate()
            elif SCENARIO == "ok":
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
                await telemetry.stop()
            elif SCENARIO in ("sigint", "sigterm"):
                fut = wait_for_signal()
                while not (WORKDIR / "ready").exists():
                    await asyncio.sleep(0.01)
                os.kill(os.getpid(), signal.SIGINT if SCENARIO == "sigint" else signal.SIGTERM)
                await fut
            if SCENARIO not in ("raise", "random_state"):
                await flow.shutdown()
        if before_exit is not None:
            result["random_unchanged_by_exit"] = random.getstate() == before_exit
    finally:
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
