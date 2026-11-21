"""Runs of many engines, one scenario per process (run by test_parts.py).

    python parts_scenarios.py <scenario> <workdir>

The run's engine hands out three parts. Two run in this process, each with an engine of its
own; the third runs in another process. A component (the results) and a check across them
(simulations against results) are registered with the run's MiniApp. Prints one JSON line
prefixed with ``RESULT ``.
"""

import asyncio
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from parts_helpers import SIMS_PER_PART, always_fails, run_part, run_part_in_process
from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from wfMiniAPI.api import MiniApp


async def main(scenario: str, workdir: Path) -> dict:
    backend = await ConcurrentExecutionBackend()
    flow = await WorkflowEngine.create(backend=backend)
    telemetry = await flow.start_telemetry(checkpoint_path=str(workdir / "launcher"))
    app = MiniApp(flow, telemetry)
    try:
        async with app, flow.workflow_scope("parts"):
            app.parameters({"parts": 3, "sims_per_part": SIMS_PER_PART})
            refs = [app.part(f"p{i}") for i in range(3)]
            ran = refs[:2] if scenario == "part_missing" else refs
            for ref in ran[:2]:
                await run_part(ref, str(workdir / "parts" / ref.name))
            if len(ran) == 3:
                ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(1, mp_context=ctx) as pool:
                    await asyncio.get_running_loop().run_in_executor(
                        pool, run_part_in_process, ran[2], str(workdir / "parts" / "p2")
                    )
            results = workdir / "results"
            results.mkdir()
            (results / "results.json").write_text(json.dumps(list(range(3 * SIMS_PER_PART))))
            app.component("results", results, verify="parts_helpers:verify_results")
            app.check("sims_match_results", verify="parts_helpers:sims_match_results")
            if scenario == "check_fails":
                app.check("always_fails", verify=always_fails)
                app.check("static", ok=True, detail="computed in the run")
    finally:
        try:
            await flow.shutdown()
        finally:
            await telemetry.stop()
            app.verify()
    return {"report": app.report.to_dict(), "run_id": flow.uid}


if __name__ == "__main__":
    out = asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
    print("RESULT " + json.dumps(out, default=str), flush=True)
