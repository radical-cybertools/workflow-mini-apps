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
spans unless `--span-ratio 1`), `services/`, `vllm/`, and `autonmi.json` (the AUTONMI commit the
run imported, against the one the example pins: `screening/pin.py`).
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
from screening.benchmark import SIM_KERNELS, Benchmark  # noqa: E402
from screening.pin import autonmi_pin  # noqa: E402

from wfMiniAPI.api import MiniApp  # noqa: E402


class _Help(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """The module's docstring as it is, and each flag's default after its help."""


def parse(argv=None) -> argparse.Namespace:
    """The command line: the campaign, the workload's knobs (`Benchmark`), the model, admission,
    the system's processes, the telemetry. Each flag's default is the knob's default."""
    d = Benchmark()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=_Help)

    g = p.add_argument_group("the campaign")
    g.add_argument(
        "--agents", type=int, default=d.agents, help="the campaign's agents, one per candidate"
    )
    g.add_argument(
        "--seed", type=int, default=d.seed, help="the seed of the candidates and every draw"
    )
    g.add_argument(
        "--populations", type=int, default=1, help="populations of agents, each with its own engine"
    )
    g.add_argument(
        "--runtime",
        choices=["local", "dragon"],
        default="local",
        help="threads, or Dragon processes",
    )
    g.add_argument(
        "--timeout", type=float, default=None, help="seconds the populations may run, all together"
    )
    g.add_argument(
        "--run-dir",
        default=None,
        help="where the run writes (default: runs/screening-<agents>-<time>)",
    )

    g = p.add_argument_group("what the agents do (the workload's knobs)")
    g.add_argument(
        "--sim-median",
        type=float,
        default=d.sim_median_s,
        help="a simulation's median duration, seconds",
    )
    g.add_argument(
        "--sim-sigma",
        type=float,
        default=d.sim_sigma,
        help="the lognormal shape of a simulation's duration",
    )
    g.add_argument(
        "--sim-distribution", choices=["lognormal", "fixed", "uniform"], default=d.sim_distribution,
        help="a simulation's duration: lognormal (median, sigma), fixed (the median) or uniform (--sim-range)",
    )  # fmt: skip
    g.add_argument(
        "--sim-range", type=float, nargs=2, metavar=("LO", "HI"), default=list(d.sim_range_s),
        help="the uniform distribution's bounds, seconds",
    )  # fmt: skip
    g.add_argument(
        "--sim-cost-sigma", type=float, default=d.sim_cost_sigma,
        help="the cost spread: the lognormal shape of a seeded cost scale per candidate (0: every candidate alike)",
    )  # fmt: skip
    g.add_argument(
        "--sims-per-agent", type=int, default=d.sims_per_agent,
        help="exactly k simulations per agent; 0 lets the physics decide (2-3)",
    )  # fmt: skip
    g.add_argument(
        "--max-sims", type=int, default=d.max_sims, help="the most simulations an agent may run"
    )
    g.add_argument(
        "--no-burn",
        dest="burn",
        action="store_false",
        help="simulations sleep instead of spinning a core",
    )
    g.add_argument(
        "--sim-out-elems", type=int, default=d.sim_out_elems,
        help="float64 values a simulation writes to the store",
    )  # fmt: skip
    g.add_argument(
        "--sim-kernel", default=d.sim_kernel, metavar="KERNEL",
        help="run this wfMiniAPI compute kernel for each simulation's duration, in place of "
        f"spinning a core: {', '.join(SIM_KERNELS)} (off by default)",
    )  # fmt: skip
    g.add_argument(
        "--sim-kernel-size", type=int, default=d.sim_kernel_size,
        help="the kernel's size: size x size matrices for matMulSimple2D, else size values",
    )  # fmt: skip
    g.add_argument(
        "--sim-kernel-device", default=d.sim_kernel_device,
        help="where the kernel runs: cpu (NumPy) or gpu (CuPy, the gpu extra)",
    )  # fmt: skip
    g.add_argument(
        "--lookup-calls", type=int, default=d.lookup_calls, help="MCP reference lookups per agent"
    )
    g.add_argument(
        "--mcp-latency-ms", type=float, nargs=2, metavar=("LO", "HI"), default=list(d.mcp_latency_ms),
        help="the reference server's time per lookup, drawn between LO and HI, milliseconds",
    )  # fmt: skip
    g.add_argument(
        "--plan-calls", type=int, default=d.plan_calls,
        help="planning calls: notes (an in-process tool) at each decision point, more model calls with no HPC work",
    )  # fmt: skip
    g.add_argument(
        "--tool-result-tokens", type=int, default=d.tool_result_tokens,
        help="long tool results: about K tokens of padding in each lookup's and each analysis's result",
    )  # fmt: skip
    g.add_argument(
        "--suspend-fraction", type=float, default=d.suspend_fraction,
        help="the share of simulations an agent suspends (saves its state and leaves memory) before",
    )  # fmt: skip
    g.add_argument("--no-suspend", action="store_true", help="never suspend: --suspend-fraction 0")
    g.add_argument(
        "--think-budget", type=int, default=d.think_budget,
        help="the thinking budget: tokens a decision call may think (0: thinking off, the knob off)",
    )  # fmt: skip
    g.add_argument(
        "--think-policy", choices=["fixed", "proportional"], default=d.think_policy,
        help="the same budget for all, or the budget times the candidate's cost scale (with a budget)",
    )  # fmt: skip
    g.add_argument(
        "--think-budget-max", type=int, default=d.think_budget_max,
        help="the clip of a proportional budget (with a budget)",
    )  # fmt: skip
    g.add_argument(
        "--think-gain", type=float, default=d.think_gain,
        help="the reasoning gain: the modeled cut in simulations that reasoning buys (0: cost only)",
    )  # fmt: skip
    g.add_argument(
        "--think-gain-on", choices=["used", "asked"], default=d.think_gain_on,
        help="the gain on the thinking tokens used, or on the budget asked (with a gain)",
    )  # fmt: skip

    g = p.add_argument_group("the model")
    g.add_argument(
        "--model", choices=["stub", "openai"], default="stub", help="the stub, or vLLM servers"
    )
    g.add_argument("--stub-latency", type=float, default=0.0, help="the stub's seconds per call")
    g.add_argument("--stub-replicas", type=int, default=1, help="stub replicas per population")
    g.add_argument(
        "--stub-kv-tokens", type=int, default=0,
        help="the stub's prefix cache, tokens per replica; it reports the prompt tokens it finds there (0: none)",
    )  # fmt: skip
    g.add_argument(
        "--stub-decode-rate", type=float, default=30.0,
        help="the stub's thinking tokens per second, with a thinking budget",
    )  # fmt: skip
    g.add_argument(
        "--stub-think-median", type=float, default=320.0,
        help="the median natural length of the stub's thinking, tokens (0: the whole budget)",
    )  # fmt: skip
    g.add_argument(
        "--stub-think-sigma", type=float, default=0.25,
        help="the lognormal shape of the stub's natural thinking length",
    )  # fmt: skip
    g.add_argument("--endpoints-file", default=None, help="the model job's base URLs, one per line")
    g.add_argument("--api-key-file", default=None, help="the model servers' key, in a file")
    g.add_argument("--model-name", default="Qwen3-8B", help="the served model's name")
    g.add_argument(
        "--max-tokens", type=int, default=512, help="answer length: the ceiling on a reply, tokens"
    )
    g.add_argument(
        "--tokenizer", default=None,
        help="with vLLM and a thinking budget: the model's tokenizer (its directory or tokenizer.json), "
        "to count thinking tokens; without it they are counted as the reasoning's share of the reply",
    )  # fmt: skip
    g.add_argument(
        "--model-wait", type=float, default=1800.0, help="seconds to wait for the model servers"
    )
    g.add_argument(
        "--vllm-interval", type=float, default=2.0, help="seconds between samples of the servers"
    )

    g = p.add_argument_group("admission: how calls reach the model")
    g.add_argument(
        "--cap-requests", "--cap", type=int, default=96,
        help="the request cap: requests in flight per replica, across the populations",
    )  # fmt: skip
    g.add_argument(
        "--cap-tokens", type=int, default=235_000,
        help="the token cap: tokens of live context in flight per replica (0: off)",
    )  # fmt: skip
    g.add_argument(
        "--reply-reserve", type=int, default=64, help="tokens reserved for each reply in admission"
    )
    g.add_argument(
        "--admission", choices=["fcfs", "cache"], default="fcfs",
        help="admission order: first come, first served (cache-first is not in this example yet)",
    )  # fmt: skip

    g = p.add_argument_group("the system's processes")
    g.add_argument("--workers", type=int, default=8, help="simulation workers per population")
    g.add_argument(
        "--reuse-event-loop", action="store_true",
        help="one event loop per worker thread for simulation tasks (needs RHAPSODY with rhapsody#163)",
    )  # fmt: skip
    g.add_argument(
        "--stagger-batch", "--start-batch", type=int, default=100, help="agents started per batch"
    )
    g.add_argument(
        "--stagger-interval", "--start-interval", type=float, default=0.5,
        help="seconds between batches of agents started",
    )  # fmt: skip
    g.add_argument("--mcp-servers", type=int, default=4, help="reference-data MCP server processes")
    g.add_argument(
        "--toolservices", type=int, default=8, help="tool services, at most one per population"
    )
    g.add_argument("--mcp-clients", type=int, default=8, help="MCP clients per tool service")
    g.add_argument("--mcp-per-client", type=int, default=16, help="calls in flight per MCP client")
    g.add_argument("--managers", type=int, default=4, help="RaDEX managers, under Dragon")
    g.add_argument(
        "--manager-gib",
        type=float,
        default=2.0,
        help="each RaDEX manager's memory, GiB (at most 2)",
    )

    g = p.add_argument_group("telemetry")
    g.add_argument(
        "--span-ratio", type=float, default=0.1,
        help="the share of agents whose spans are exported (1: every span)",
    )  # fmt: skip
    g.add_argument(
        "--metrics-interval", type=float, default=10.0, help="seconds between metric exports"
    )

    cfg = p.parse_args(argv)
    if cfg.no_suspend:
        cfg.suspend_fraction = 0.0
    if cfg.admission != "fcfs":
        p.error("--admission cache: cache-first admission is not in this example yet")
    return cfg


def benchmark(cfg: argparse.Namespace) -> Benchmark:
    """The workload's knobs, from the command line."""
    return Benchmark(
        agents=cfg.agents,
        seed=cfg.seed,
        max_sims=cfg.max_sims,
        sims_per_agent=cfg.sims_per_agent,
        lookup_calls=cfg.lookup_calls,
        sim_distribution=cfg.sim_distribution,
        sim_median_s=cfg.sim_median,
        sim_sigma=cfg.sim_sigma,
        sim_range_s=tuple(cfg.sim_range),
        sim_out_elems=cfg.sim_out_elems,
        sim_burn=cfg.burn,
        sim_kernel=cfg.sim_kernel,
        sim_kernel_size=cfg.sim_kernel_size,
        sim_kernel_device=cfg.sim_kernel_device,
        mcp_latency_ms=tuple(cfg.mcp_latency_ms),
        suspend_fraction=cfg.suspend_fraction,
        sim_cost_sigma=cfg.sim_cost_sigma,
        think_budget=cfg.think_budget,
        think_policy=cfg.think_policy,
        think_budget_max=cfg.think_budget_max,
        think_gain=cfg.think_gain,
        think_gain_on=cfg.think_gain_on,
        plan_calls=cfg.plan_calls,
        tool_result_tokens=cfg.tool_result_tokens,
    )


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
    bench = benchmark(cfg)
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
        Screening(
            bench,
            stub_latency_s=cfg.stub_latency,
            stub_kv_tokens=cfg.stub_kv_tokens,
            stub_decode_rate=cfg.stub_decode_rate,
            stub_think_median=cfg.stub_think_median,
            stub_think_sigma=cfg.stub_think_sigma,
            max_tokens=cfg.max_tokens,
            tokenizer=cfg.tokenizer,
        ),
        CampaignConfig(
            run_dir=str(run_dir),
            populations=cfg.populations,
            runtime=cfg.runtime,
            mcp_servers=cfg.mcp_servers,
            toolservices=cfg.toolservices,
            mcp_clients=cfg.mcp_clients,
            mcp_per_client=cfg.mcp_per_client,
            store_managers=cfg.managers,
            store_manager_gib=cfg.manager_gib,
            telemetry=Telemetry(span_ratio=cfg.span_ratio, interval_s=cfg.metrics_interval),
            timeout_s=cfg.timeout,
            population=PopulationConfig(
                workers=cfg.workers,
                stagger_batch=cfg.stagger_batch,
                stagger_interval_s=cfg.stagger_interval,
                reuse_event_loop=cfg.reuse_event_loop,
                cap_tokens=cfg.cap_tokens,
                cap_requests=cfg.cap_requests,
                reply_reserve=cfg.reply_reserve,
                stub_replicas=cfg.stub_replicas,
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

            sampler = VllmSampler(list(model.endpoints), api_key(model), cfg.vllm_interval)
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
    (run_dir / "autonmi.json").write_text(json.dumps(autonmi_pin()) + "\n")
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
