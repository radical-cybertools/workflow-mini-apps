"""What the part scenarios import by name: a part's engine, a component verifier, checks.

Importable by name, so that the offline verifier can run them again from the files.
"""

import asyncio
import json
from pathlib import Path

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from wfMiniAPI.api import MiniApp
from wfMiniAPI.telemetry.run import ComponentReport

SIMS_PER_PART = 3


async def run_part(ref, checkpoint_dir: str, sims: int = SIMS_PER_PART) -> dict:
    """One part of the run: an engine of its own, its telemetry, and its own MiniApp."""
    backend = await ConcurrentExecutionBackend()
    flow = await WorkflowEngine.create(backend=backend)
    telemetry = await flow.start_telemetry(checkpoint_path=checkpoint_dir)
    app = MiniApp(flow, telemetry, part_of=ref)

    @flow.function_task
    async def sim(i):
        return i * i

    try:
        async with app, flow.workflow_scope(ref.name):
            await asyncio.gather(*(sim(i) for i in range(sims)))
    finally:
        try:
            await flow.shutdown()
        finally:
            await telemetry.stop()
            app.verify()
    return app.report.to_dict()


def run_part_in_process(ref, checkpoint_dir: str) -> dict:
    """The same, in a process of its own: only ``ref`` and a path cross the boundary."""
    return asyncio.run(run_part(ref, checkpoint_dir))


def verify_results(files) -> ComponentReport:
    path = Path(files) / "results.json"
    if not path.exists():
        return ComponentReport(False, reasons=(f"{path.name} missing",))
    return ComponentReport(True, counts={"results": len(json.loads(path.read_text()))})


def sims_match_results(run):
    sims = sum(p["completed_by_executable"].get("sim", 0) for p in run.parts.values())
    results = run.components["results"]["counts"].get("results")
    return sims == results, {"sims": sims, "results": results}


def always_fails(run):
    return False, "on purpose"
