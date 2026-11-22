"""The synthetic screening benchmark as a workflow mini-app: an AsyncFlow application of many
engines, whose agents run on the AUTONMI framework.

    python examples/agentic-screening/miniapp_screening.py --agents 200 --populations 2 \
        --sim-median 0.05 --no-burn

    # under Dragon: RaDEX, tool services and populations as Dragon processes
    python -m dragon -s examples/agentic-screening/miniapp_screening.py --runtime dragon \
        --populations 2 --workers 52 --model openai --endpoints-file FLEET/endpoints \
        --api-key-file FLEET/api-key

The script has workflow-mini-apps' shape: it builds the engine and its telemetry, attaches
`wfMiniAPI.api.MiniApp`, runs the workflow inside it, and owns the shutdown. The workflow adds the
campaign's populations to the engine (one AsyncFlow node each, with an engine of their own for
their simulations), then ranks the candidates as an ordinary AsyncFlow task.

With `--runtime dragon` the mini-app's engine runs on RHAPSODY's native Dragon backend (2 pool
workers), and each population is a function task that backend starts as a Dragon process of its
own. With `--model openai` the fleet calls a model job's vLLM servers (temperature 0, 512 tokens,
thinking off), and the mini-app samples the servers' own counters over the campaign, so the run's
checks can compare Flowgentic's token counts with the servers'.

The application, `screening/` beside this script, is put first on `PYTHONPATH`, so that the
processes the run starts (the MCP servers; under Dragon, the populations and their workers) import
it too.

Writes `<run-dir>/`: `launcher/` (the mini-app's checkpoint and MiniApp's report), `populations/`
(each population's checkpoint and stats), `app/` (per-agent rows, checks, ranking), `otel/`
(Flowgentic's export, one directory per population process under Dragon; a tenth of the agents'
spans unless `--span-ratio 1`), `services/`, `vllm/`.
Exit status 0 when every check passes and the telemetry is complete.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH")]))

from autonmi import Campaign, CampaignConfig  # noqa: E402 - after the application's path
from autonmi.models import api_key, wait_for_servers  # noqa: E402
from autonmi.population import PopulationConfig  # noqa: E402
from autonmi.telemetry import Telemetry  # noqa: E402
from radical.asyncflow import WorkflowEngine  # noqa: E402
from screening.agent import Screening  # noqa: E402
from screening.benchmark import Benchmark  # noqa: E402

from wfMiniAPI.api import MiniApp  # noqa: E402


def parse(argv=None) -> argparse.Namespace:
    d = Benchmark()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--agents", type=int, default=d.agents)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--populations", type=int, default=1)
    p.add_argument("--workers", type=int, default=8, help="simulation workers per population")
    p.add_argument(
        "--reuse-event-loop",
        action="store_true",
        help="one event loop per worker thread for simulation tasks (needs RHAPSODY with rhapsody#163)",
    )
    p.add_argument("--runtime", choices=["local", "dragon"], default="local")
    p.add_argument("--mcp-servers", type=int, default=4)
    p.add_argument("--toolservices", type=int, default=8, help="at most one per population")
    p.add_argument("--lookup-calls", type=int, default=d.lookup_calls)
    p.add_argument("--sims-per-agent", type=int, default=d.sims_per_agent)
    p.add_argument("--sim-median", type=float, default=d.sim_median_s)
    p.add_argument("--suspend-fraction", type=float, default=d.suspend_fraction)
    p.add_argument("--no-burn", dest="burn", action="store_false")
    p.add_argument("--stub-latency", type=float, default=0.0)
    p.add_argument("--stagger-batch", type=int, default=100)
    p.add_argument("--stagger-interval", type=float, default=0.5)
    p.add_argument("--model", choices=["stub", "openai"], default="stub")
    p.add_argument("--endpoints-file", default=None, help="the model job's base URLs, one per line")
    p.add_argument("--api-key-file", default=None)
    p.add_argument("--model-name", default="Qwen3-8B")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--cap-tokens", type=int, default=235_000, help="admitted tokens per replica")
    p.add_argument("--cap-requests", type=int, default=96, help="admitted requests per replica")
    p.add_argument(
        "--model-wait", type=float, default=1800.0, help="seconds to wait for the model servers"
    )
    p.add_argument(
        "--vllm-interval", type=float, default=2.0, help="seconds between samples of the servers"
    )
    p.add_argument(
        "--span-ratio",
        type=float,
        default=0.1,
        help="share of the agents whose spans are exported (1: every span)",
    )
    p.add_argument(
        "--metrics-interval", type=float, default=10.0, help="seconds between metric exports"
    )
    p.add_argument(
        "--timeout", type=float, default=None, help="seconds the populations may run, all together"
    )
    p.add_argument("--run-dir", default=None)
    return p.parse_args(argv)


async def make_backend(cfg: argparse.Namespace):
    """The mini-app's own backend: RHAPSODY's native one under Dragon, a thread pool locally."""
    if cfg.runtime == "dragon":
        from rhapsody.backends import DragonNativeExecutionBackend

        return await DragonNativeExecutionBackend(workers_per_node=2)
    from rhapsody.backends import ConcurrentExecutionBackend

    return await ConcurrentExecutionBackend(ThreadPoolExecutor(max_workers=4))


async def run_miniapp(cfg: argparse.Namespace, run_dir: Path) -> int:
    logger = logging.getLogger("screening")
    backend = await make_backend(cfg)
    flow = await WorkflowEngine.create(backend=backend)
    telemetry = await flow.start_telemetry(checkpoint_path=str(run_dir / "launcher"))
    app = MiniApp(flow, telemetry)
    ok = False
    try:
        async with app, flow.workflow_scope("screening"):
            ok = await miniapp_workflow(cfg, flow, run_dir, logger, app)
    finally:
        try:
            await flow.shutdown()
        finally:
            await telemetry.stop()
            app.verify()
    report = app.report
    print(
        f"[screening] checks {'pass' if ok and report.checks_pass else 'FAIL'}; telemetry "
        f"{'complete' if report.telemetry_complete else 'INCOMPLETE'}; {run_dir}",
        flush=True,
    )
    return 0 if ok and report.exit_hint == 0 else 1


async def miniapp_workflow(cfg, flow, run_dir: Path, logger, app) -> bool:
    bench = Benchmark(
        agents=cfg.agents,
        seed=cfg.seed,
        lookup_calls=cfg.lookup_calls,
        sims_per_agent=cfg.sims_per_agent,
        sim_median_s=cfg.sim_median,
        sim_burn=cfg.burn,
        suspend_fraction=cfg.suspend_fraction,
    )
    model = None
    if cfg.model == "openai":  # the model job's servers, once all answer: the fleet's settings
        model = await wait_for_servers(
            cfg.endpoints_file,
            cfg.api_key_file,
            wait_s=cfg.model_wait,
            name=cfg.model_name,
            max_tokens=cfg.max_tokens,
        )
    campaign = Campaign(
        flow,
        Screening(bench, stub_latency_s=cfg.stub_latency),
        CampaignConfig(
            run_dir=str(run_dir),
            populations=cfg.populations,
            runtime=cfg.runtime,
            mcp_servers=cfg.mcp_servers,
            toolservices=cfg.toolservices,
            telemetry=Telemetry(span_ratio=cfg.span_ratio, interval_s=cfg.metrics_interval),
            timeout_s=cfg.timeout,
            population=PopulationConfig(
                workers=cfg.workers,
                stagger_batch=cfg.stagger_batch,
                stagger_interval_s=cfg.stagger_interval,
                reuse_event_loop=cfg.reuse_event_loop,
                cap_tokens=cfg.cap_tokens,
                cap_requests=cfg.cap_requests,
                model=model,
            ),
        ),
        app=app,
    )
    try:
        await campaign.start_services()
        sampler = None
        if model is not None:
            from autonmi.vllm import VllmSampler

            sampler = VllmSampler(model["endpoints"], api_key(model), cfg.vllm_interval)
            sampler.start()
        try:
            t0 = time.time()
            await campaign.run()
            logger.info("campaign done in %.1f s", time.time() - t0)
        finally:
            if sampler is not None:
                await asyncio.to_thread(sampler.stop)
                (run_dir / "vllm").mkdir(parents=True, exist_ok=True)
                (run_dir / "vllm" / "vllm.json").write_text(json.dumps(sampler.raw()) + "\n")
            await campaign.stop_services()

        @flow.function_task
        async def rank_candidates(store_spec: str, agents: int) -> list:
            from autonmi.store import open_store
            from screening.verify import rank, read_reports

            return rank(read_reports(open_store(store_spec), agents))

        ranking = await rank_candidates(campaign.store_spec, bench.agents)
        (run_dir / "app" / "ranking.json").write_text(json.dumps(ranking, indent=1) + "\n")
        return campaign.check()
    finally:
        await campaign.close()


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    cfg = parse(argv)
    if cfg.model == "openai" and not cfg.endpoints_file:
        raise SystemExit("--model openai needs --endpoints-file")
    run_dir = Path(cfg.run_dir or f"runs/screening-{cfg.agents}-{time.strftime('%Y%m%dT%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    from autonmi.assemble import assemble

    if float(os.environ.get("AUTONMI_STACKS") or 0) > 0:  # every thread's stack, every s seconds
        import faulthandler

        stacks = open(run_dir / "launcher-stacks.txt", "w")  # noqa: SIM115
        faulthandler.dump_traceback_later(
            float(os.environ["AUTONMI_STACKS"]), repeat=True, file=stacks
        )
    rc = asyncio.run(
        run_miniapp(cfg, run_dir)
    )  # the campaign configures Flowgentic's export, and shuts it down
    summary = assemble(run_dir)
    print(
        f"[screening] assembled: {summary['agents']} agents, {summary['simulations']} simulations, "
        f"{summary['model_calls']} model calls; checks {'pass' if summary['ok'] else 'FAIL'}",
        flush=True,
    )
    return rc or (0 if summary["ok"] else 1)


if __name__ == "__main__":
    sys.exit(main())
