"""MiniApp scenarios on the Concurrent backend, one per process (run by test_miniapp.py).

    python scenarios.py <scenario> <workdir>

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
from rhapsody.backends import ConcurrentExecutionBackend

from wfMiniAPI.api import AttachError, MiniApp

SCENARIO, WORKDIR = sys.argv[1], Path(sys.argv[2])
result = {"scenario": SCENARIO}


def done(**fields):
    result.update(fields)
    print("RESULT " + json.dumps(result, default=str), flush=True)


async def engine():
    flow = await WorkflowEngine.create(backend=await ConcurrentExecutionBackend())
    return flow


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
            return done(attach_error=str(exc))
        return done(attach_error=None)

    if SCENARIO == "wrong_telemetry":
        other = await engine()
        try:
            async with MiniApp(other, telemetry):
                pass
        except AttachError as exc:
            return done(attach_error=str(exc))
        return done(attach_error=None)

    app = MiniApp(flow, telemetry)
    t0 = time.monotonic()
    try:
        async with app, flow.workflow_scope("scenario"):
            if SCENARIO == "random_state":
                result["random_unchanged_in_block"] = True
                before = random.getstate()
                await asyncio.sleep(0.2)
                result["random_unchanged_in_block"] = random.getstate() == before
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
            if SCENARIO not in ("raise",):
                await flow.shutdown()
    finally:
        report = app.report.to_dict() if app.report else None
        done(report=report, elapsed=time.monotonic() - t0)


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
