"""The synthetic screening benchmark's agents, as an agent definition for the AUTONMI framework.

One agent per candidate material. Each agent looks the material up (an MCP tool), simulates (a task
on its population's AsyncFlow engine; it suspends while the task runs), analyzes the result (inline)
and repeats until the analysis converges, then reports (inline). The scoring is seeded, so every
agent's best is known and the run verifies itself.

The workload is the benchmark's: its knobs (`benchmark.Benchmark`), the physics, the prompt, the
simulation's body and the verification, as the benchmark's original LangGraph agents run them. Here
the agent runs on Flowgentic's loop (OpenAI-format messages), the stub model is a Flowgentic
transport, and each simulation is an AsyncFlow task. Not yet here: notes (`plan_calls`), the
reasoning knob (`think_budget`), the nudge to an agent that ends without a report, and longer tool
results.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from autonmi.agent import AgentSpec, Check, PopulationContext, ToolAgent
from flowgentic.host import current_agent  # the agent an inline tool runs for
from flowgentic.tools import TaskRecord, mcp_tool, task_tool
from pydantic import BaseModel, Field

from . import physics
from .benchmark import Benchmark
from .prompts import system_prompt, task_message
from .sim import run_simulation
from .stub import ScreeningStub
from .tools import Budget, SimulateArgs
from .verify import read_reports, verify

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


class LookupArgs(BaseModel):
    material: str = Field(description="the material's name, such as mat-000123")


LOOKUP = mcp_tool(
    "lookup_reference",
    description="Look up reference data for a material: a related compound's synthesis conditions and figure of merit.",
    args_schema=LookupArgs,
    idempotent=True,
)


@dataclass
class Record:
    """What the application keeps of one agent."""

    cid: str
    sims: list = field(default_factory=list)  # (temperature, doping, result key)
    asked: int = 0
    analyses: int = 0
    lookups: int = 0
    store_misses: int = 0
    wrong_keys: list = field(default_factory=list)
    report: dict | None = None


class Screening:
    """The agent definition of the screening benchmark."""

    name = "screening"

    def __init__(
        self, bench: Benchmark, stub_latency_s: float = 0.0, stub_tool_retries: int = 3
    ) -> None:
        self.knobs = bench
        self.stub_latency_s = stub_latency_s
        self.stub_tool_retries = stub_tool_retries

    # --- the campaign's agents and services ------------------------------------------------------

    def agents(self):
        bench = self.knobs
        for i in range(bench.agents):
            c = physics.candidate(i, bench)
            yield AgentSpec(c.cid, [{"role": "user", "content": task_message(c, bench)}])

    def services(self, run_dir: Path, servers: int = 1) -> tuple[list, list[str]]:
        if not self.knobs.lookup_calls:
            return [], []
        from .mcp_server import start_servers

        run_dir.mkdir(parents=True, exist_ok=True)
        return start_servers(servers, self.knobs, run_dir)

    def stub(self, replica: int) -> ScreeningStub:
        return ScreeningStub(self.knobs, self.stub_latency_s, self.stub_tool_retries)

    # --- one population's agent ------------------------------------------------------------------

    def build(self, pop: PopulationContext) -> ToolAgent:
        bench = self.knobs
        records = {cid: Record(cid) for cid in pop.agent_ids}
        candidates = {cid: physics.candidate(physics.index_of(cid), bench) for cid in pop.agent_ids}
        pop.state["records"] = records

        def prepare(agent_id: str, call_id: str, args: dict) -> dict:
            rec = records[agent_id]
            rec.asked += 1
            return {
                "store_spec": pop.store_spec,
                "key": f"res/{agent_id}/{_UNSAFE.sub('_', call_id)}",
                "index": physics.index_of(agent_id),
                "temperature": float(args["temperature"]),
                "doping": float(args["doping"]),
                "bench": bench,
            }

        def suspend(agent_id: str, args: dict) -> bool:
            # Flowgentic's loop asks before it submits (its LangGraph path asks after `prepare`): this
            # is the agent's n-th simulation, as the benchmark counts it.
            return physics.suspends(agent_id, records[agent_id].asked + 1, bench)

        simulate = task_tool(
            run_simulation,
            name="simulate",
            description=SimulateArgs.__doc__,
            args_schema=SimulateArgs,
            prepare=prepare,
            output=lambda v: {"result_key": v["key"], "seconds": round(v["seconds"], 3)},
            suspend=suspend,
        )

        def on_task(r: TaskRecord) -> None:
            if not r.ok:
                return
            v, rec = r.value, records.get(r.info.agent_id)
            if rec is not None and not any(s[2] == v["key"] for s in rec.sims):
                rec.sims.append((r.info.kwargs["temperature"], r.info.kwargs["doping"], v["key"]))

        async def analyze(result_key: str) -> str:
            """Analyze a simulation's result: its score, whether it has converged, and the next conditions to try."""
            cid = current_agent()
            rec = records[cid]
            written = {s[2] for s in rec.sims}
            try:
                if result_key not in written:
                    raise KeyError(
                        f"{result_key} is not a result_key that simulate returned for {cid}"
                    )
                tensor = pop.store.get_tensor(result_key)
            except KeyError as exc:
                if result_key in written:
                    rec.store_misses += 1
                    return json.dumps({"error": f"no result under {result_key!r}: {exc}"})
                rec.wrong_keys.append(str(result_key)[:160])
                keys = [k for _, _, k in rec.sims]
                return json.dumps(
                    {
                        "error": f"{result_key!r} is not a result_key that simulate returned.",
                        "result_keys_returned_by_simulate": keys,
                        "hint": "Call analyze with one of these keys, copied exactly."
                        if keys
                        else "Call simulate first.",
                    }
                )
            c = candidates[cid]
            temperature, doping = round(float(tensor[0]), 3), round(float(tensor[1]), 5)
            s = float(tensor[2:].max())
            n = 1 + [k for _, _, k in rec.sims].index(result_key)
            _, done, nxt = physics.analysis(c, temperature, doping, n, bench, 0)
            out = {
                "result_key": result_key,
                "temperature": temperature,
                "doping": doping,
                "score": round(s, 4),
                "converged": done,
            }
            if nxt is not None:
                out["next"] = {"temperature": nxt[0], "doping": nxt[1]}
            rec.analyses += 1
            return json.dumps(out)

        async def report(temperature: float, doping: float, score: float) -> str:
            """Report the final answer: the best synthesis temperature, dopant fraction and score found."""
            cid = current_agent()
            rec = records[cid]
            c = candidates[cid]
            value = {
                "material": cid,
                "temperature": float(temperature),
                "doping": float(doping),
                "score_reported": float(score),
                "score": physics.score(c, float(temperature), float(doping)),
                "simulations": len(rec.sims),
                "t": time.time(),
            }
            pop.store.put(f"report/{cid}", value)
            rec.report = value
            return json.dumps({"saved": f"report/{cid}"})

        tools = [simulate, analyze, report]
        if bench.lookup_calls:
            tools.insert(0, LOOKUP)
        return ToolAgent(
            system_prompt=system_prompt(bench),
            tools=tools,
            max_model_calls=bench.lookup_calls + 4 * bench.max_sims + 10,
            order=Budget(bench.max_sims),
            on_task=on_task,
        )

    # --- results and checks ----------------------------------------------------------------------

    def rows(self, pop: PopulationContext) -> list[dict]:
        return [
            {
                "agent_id": rec.cid,
                "cid": rec.cid,
                "sims": rec.sims,
                "simulations": len(rec.sims),
                "analyses": rec.analyses,
                "store_misses": rec.store_misses,
                "wrong_keys": rec.wrong_keys,
                "report": rec.report,
                "finished": rec.report is not None,
                "population": pop.index,
            }
            for rec in pop.state["records"].values()
        ]

    def verify(self, rows: list[dict], store) -> list[Check]:
        bench = self.knobs
        for row in rows:  # what the verification reads: an error only for an agent that failed
            if row.get("status") not in (None, "done") and not row.get("error"):
                row["error"] = row["status"]
        v = verify(bench, read_reports(store, bench.agents), rows)
        calls = sum(r.get("model_calls", 0) for r in rows)
        scripted = sum(bench.model_calls_per_agent(len(r["sims"])) for r in rows)
        return [
            Check("every_agent_finishes_and_verifies", v["verified"] == bench.agents, v),
            Check(
                "model_calls_as_scripted",
                calls == scripted,
                {"model_calls": calls, "scripted": scripted},
            ),
            Check(
                "tensors_to_store_and_back",
                all(r["analyses"] >= 1 for r in rows if r["sims"])
                and not any(r["store_misses"] for r in rows),
                None,
            ),
        ]
