"""The synthetic screening benchmark's agents, as an agent definition for the AUTONMI framework.

One agent per candidate material. Each agent looks the material up (an MCP tool), simulates (a task
on its population's AsyncFlow engine; it suspends while the task runs), analyzes the result (inline)
and repeats until the analysis converges, then reports (inline). The scoring is seeded, so every
agent's best is known and the run verifies itself.

The workload is the benchmark's: its knobs (`benchmark.Benchmark`), the physics, the prompt, the
simulation's body and the verification, as the benchmark's original LangGraph agents run them. Here
the agent runs on Flowgentic's loop (OpenAI-format messages), the stub model is a Flowgentic
transport, and each simulation is an AsyncFlow task. With `plan_calls` the agent has a `note` tool
(in its process), and with `tool_result_tokens` lookups and analyses return that much more text.
The reasoning knob (`think_budget`): a decision call (`stub.decision_call`) gets the agent's
thinking budget (`physics.think_budget`) through the loop's `model_kwargs`, as vLLM takes it, with
thinking on and room in `max_tokens`. For each simulation the reply asks for, the agent keeps the
budget the gain counts (`physics.gain_budget`: the budget, or the thinking tokens used if fewer),
and the analysis of that simulation's result reads it. On vLLM, whose chat API does not count the
thinking tokens apart, they are counted from the reasoning text: with a tokenizer when given,
otherwise as the reasoning's share of the reply's characters.

An agent that ends without calling `report` (Qwen3-8B now and then answers in text after a forced
end of thinking) is told once to report (`prompts.NUDGE`), as the benchmark's agents are. An agent
that passes `analyze` a key `simulate` did not return gets the keys it may use and calls again. Each
nudge and each wrong key is one model call more than the script's, and the check of the calls
counts both.
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
from .prompts import NUDGE, system_prompt, task_message
from .sim import run_simulation
from .stub import ScreeningStub, decision_call
from .tools import Budget, SimulateArgs
from .verify import overruns, read_reports, verify

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
    sim_seconds: list = field(default_factory=list)  # [asked, taken] for each of `sims`
    asked: int = 0
    analyses: int = 0
    lookups: int = 0
    store_misses: int = 0
    wrong_keys: list = field(default_factory=list)
    report: dict | None = None
    notes: int = 0
    note_k: int = 0  # notes at the current decision point
    call_budgets: dict = field(
        default_factory=dict
    )  # simulate call id -> the budget its gain counts
    sim_budgets: dict = field(default_factory=dict)  # result key -> the same, for its analysis
    thinking_tokens: int = 0
    think_calls: list = field(default_factory=list)  # each decision call with a budget, in order
    nudges: int = 0  # times told to report after ending without a report
    kernel_calls: int = 0  # with a simulation kernel: its calls, over the agent's simulations


class Screening:
    """The agent definition of the screening benchmark."""

    name = "screening"

    def __init__(
        self,
        bench: Benchmark,
        stub_latency_s: float = 0.0,
        stub_tool_retries: int = 3,
        stub_kv_tokens: int = 0,
        stub_decode_rate: float = 30.0,
        stub_think_median: float = 320.0,
        stub_think_sigma: float = 0.25,
        max_tokens: int = 512,
        thinking_on: dict | None = None,
        tokenizer: str | None = None,
    ) -> None:
        self.knobs = bench
        self.stub_latency_s = stub_latency_s
        self.stub_tool_retries = stub_tool_retries
        self.stub_kv_tokens = stub_kv_tokens
        self.stub_decode_rate = stub_decode_rate
        self.stub_think_median = stub_think_median
        self.stub_think_sigma = stub_think_sigma
        self.max_tokens = max_tokens  # a reply's ceiling, as the fleet's settings give it
        self.thinking_on = dict(thinking_on or {"enable_thinking": True})  # Qwen3's switch
        self.tokenizer = (
            tokenizer  # a model directory or its tokenizer.json, to count thinking tokens
        )

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
        return ScreeningStub(
            self.knobs,
            self.stub_latency_s,
            self.stub_tool_retries,
            self.stub_kv_tokens,
            decode_rate=self.stub_decode_rate,
            think_median=self.stub_think_median,
            think_sigma=self.stub_think_sigma,
        )

    # --- one population's agent ------------------------------------------------------------------

    def build(self, pop: PopulationContext) -> ToolAgent:
        bench = self.knobs
        records = {cid: Record(cid) for cid in pop.agent_ids}
        candidates = {cid: physics.candidate(physics.index_of(cid), bench) for cid in pop.agent_ids}
        pop.state["records"] = records

        def prepare(agent_id: str, call_id: str, args: dict) -> dict:
            rec = records[agent_id]
            rec.asked += 1
            rec.note_k = 0  # a simulation: the decision point's notes are done
            key = f"res/{agent_id}/{_UNSAFE.sub('_', call_id)}"
            budget = rec.call_budgets.pop(call_id, 0)  # of the decision call that chose it
            if budget:
                rec.sim_budgets[key] = budget
            return {
                "store_spec": pop.store_spec,
                "key": key,
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
                rec.sim_seconds.append([round(v["seconds"], 4), round(v["seconds_taken"], 4)])
                rec.kernel_calls += v.get("kernel_calls", 0)

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
            budget = rec.sim_budgets.get(
                result_key, 0
            )  # of the decision call that chose its conditions
            _, done, nxt = physics.analysis(c, temperature, doping, n, bench, budget)
            out = {
                "result_key": result_key,
                "temperature": temperature,
                "doping": doping,
                "score": round(s, 4),
                "converged": done,
            }
            if nxt is not None:
                out["next"] = {"temperature": nxt[0], "doping": nxt[1]}
            if bench.tool_result_tokens:
                out["spectrum"] = physics.spectrum_sample(
                    tensor[2:], bench.tool_result_tokens, f"{bench.seed}/{result_key}"
                )
            rec.analyses += 1
            rec.note_k = 0  # a new decision point
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

        async def note(text: str) -> str:
            """Record a note in your notebook: your plan, or your reflection on a result."""
            rec = records[current_agent()]
            rec.notes += 1
            rec.note_k += 1
            out = {"saved": f"note {rec.note_k} of {bench.plan_calls}"}
            if rec.report is not None:  # the work is done: a model can add closing notes
                out["hint"] = "Your report is saved. Reply with the single word DONE."
            elif rec.note_k > bench.plan_calls:  # at temperature 0 a model can keep noting
                out["hint"] = (
                    "The notes for this step are written. Go on with the next step: "
                    "call simulate or report."
                )
            return json.dumps(out)

        tools = [simulate, analyze, report]
        if bench.lookup_calls:
            tools.insert(0, LOOKUP)
        if bench.plan_calls:
            tools.append(note)
        tokenizer = _tokenizer(self.tokenizer) if bench.think_budget and self.tokenizer else None
        thinking_on, max_tokens = self.thinking_on, self.max_tokens

        def model_kwargs(agent_id: str, messages: list[dict]) -> dict:
            """A decision call's thinking budget, as vLLM takes it; the other calls nothing."""
            if not bench.think_budget or not decision_call(messages, bench):
                return {}
            b = physics.think_budget(physics.index_of(agent_id), bench)
            if not b:
                return {}
            return {
                "extra_body": {
                    "chat_template_kwargs": dict(thinking_on),
                    "thinking_token_budget": b,
                },
                "max_tokens": max_tokens + b,
                "reserve": b,  # admission counts the thinking, which stays out of the context
                "meta": {"think_budget": b},
            }

        def on_call(call, reply) -> None:
            """A decision call's reply: the budget its gain counts, kept for each simulation it asks for."""
            b = (call.meta or {}).get("think_budget") if isinstance(call.meta, dict) else None
            if not b:
                return
            rec = records[call.agent_id]
            used = thinking_tokens(reply, tokenizer)
            rec.thinking_tokens += used
            rec.think_calls.append(
                {
                    "budget": b,
                    "thinking": used,
                    "completion_tokens": reply.usage.completion_tokens,
                    "reasoning_chars": len(reply.reasoning or ""),
                    "tool_calls": len(reply.tool_calls),
                }
            )
            gain = physics.gain_budget(b, used, bench)
            for c in reply.tool_calls:
                if c.name == "simulate" and gain:
                    rec.call_budgets[c.id] = gain

        def nudge(agent_id: str) -> str | None:
            rec = records[agent_id]
            if rec.report is not None:
                return None
            rec.nudges += 1
            return NUDGE

        return ToolAgent(
            system_prompt=system_prompt(bench),
            tools=tools,
            max_model_calls=bench.lookup_calls
            + 4 * bench.max_sims
            + 10
            + bench.plan_calls * (bench.max_sims + 1),
            order=Budget(bench.max_sims),
            on_task=on_task,
            on_call=on_call,
            model_kwargs=model_kwargs,
            nudge=nudge,
        )

    # --- results and checks ----------------------------------------------------------------------

    def rows(self, pop: PopulationContext) -> list[dict]:
        bench = self.knobs
        return [
            {
                "agent_id": rec.cid,
                "cid": rec.cid,
                "sims": rec.sims,
                "sim_seconds": rec.sim_seconds,
                "simulations": len(rec.sims),
                "analyses": rec.analyses,
                "store_misses": rec.store_misses,
                "wrong_keys": rec.wrong_keys,
                "report": rec.report,
                "finished": rec.report is not None,
                "population": pop.index,
                **({"notes": rec.notes} if bench.plan_calls else {}),
                **({"nudges": rec.nudges} if rec.nudges else {}),
                **({"kernel_calls": rec.kernel_calls} if bench.sim_kernel else {}),
                **(
                    {
                        "think_budget": physics.think_budget(physics.index_of(rec.cid), bench),
                        "thinking_tokens": rec.thinking_tokens,
                        "think_calls": rec.think_calls,
                        "cost_scale": round(
                            physics.cost_scale(physics.index_of(rec.cid), bench), 4
                        ),
                    }
                    if bench.think_budget or bench.sim_cost_sigma
                    else {}
                ),
            }
            for rec in pop.state["records"].values()
        ]

    def verify(self, rows: list[dict], store) -> list[Check]:
        bench = self.knobs
        for row in rows:  # what the verification reads: an error only for an agent that failed
            if row.get("status") not in (None, "done") and not row.get("error"):
                row["error"] = row["status"]
        v = verify(bench, read_reports(store, bench.agents), rows)
        seconds = overruns(rows)
        calls = sum(r.get("model_calls", 0) for r in rows)
        nudges = sum(r.get("nudges", 0) for r in rows)
        wrong_keys = sum(len(r.get("wrong_keys") or ()) for r in rows)
        scripted = sum(bench.model_calls_per_agent(len(r["sims"])) for r in rows)
        scripted += nudges + wrong_keys
        return [
            Check("every_agent_finishes_and_verifies", v["verified"] == bench.agents, v),
            Check(
                "model_calls_as_scripted",
                calls == scripted,
                {
                    "model_calls": calls,
                    "scripted": scripted,
                    "nudges": nudges,
                    "wrong_keys": wrong_keys,
                },
            ),
            Check(
                "tensors_to_store_and_back",
                all(r["analyses"] >= 1 for r in rows if r["sims"])
                and not any(r["store_misses"] for r in rows),
                None,
            ),
            # a simulation's seconds are its cost, not a deadline: overruns are reported, not failed
            Check("simulations_ran_their_seconds", seconds["ended_early"] == 0, seconds),
        ]


def thinking_tokens(reply, tokenizer=None) -> int:
    """Thinking tokens of a reply: the server's count when it gives one; else the reasoning text's
    tokens with a tokenizer; else the reasoning's share of the reply's characters (with its tool
    calls), times the completion tokens. vLLM's chat API does not count them apart."""
    usage = reply.usage
    if usage.reasoning_tokens is not None:
        return int(usage.reasoning_tokens)
    text = reply.reasoning or ""
    if not text:
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)
    visible = len(reply.content or "") + sum(
        len(c.name) + len(json.dumps(c.args)) for c in reply.tool_calls
    )
    completion = usage.completion_tokens or 0
    return round(completion * len(text) / (len(text) + visible)) if completion else 0


def _tokenizer(path: str):
    """A Hugging Face fast tokenizer from a model directory or its `tokenizer.json`."""
    import os

    from tokenizers import Tokenizer

    return Tokenizer.from_file(
        os.path.join(path, "tokenizer.json") if os.path.isdir(path) else path
    )
