"""The screening stub as a Flowgentic transport: it plays the workflow from OpenAI-format messages.

It makes the decisions of the benchmark's own stub, read from Flowgentic's loop's messages (dicts,
tool results named through their call's id), so a run without GPUs keeps the workflow's shape and
its per-agent results equal those of the benchmark's original agents. Usage is counted as
characters / 4. Not yet here: notes (`plan_calls`), thinking (`think_budget`) and the prefix cache.
"""

from __future__ import annotations

import asyncio
import json
import re

from flowgentic.fleet import Reply, ToolCall, Usage

from . import physics
from .benchmark import Benchmark

CID = re.compile(r"mat-\d{6}")


class ToolGaveUp(RuntimeError):
    """A tool kept failing after the stub's retries."""


def _data(content) -> dict:
    try:
        d = json.loads(content)
    except (TypeError, ValueError):
        return {"error": str(content)}
    return d if isinstance(d, dict) else {"error": str(d)}


def tool_results(messages: list[dict]) -> list[tuple[str, dict]]:
    """Every tool result of a conversation, in order, as (tool name, data)."""
    names: dict[str, str] = {}
    out = []
    for m in messages:
        if m.get("role") == "assistant":
            for call in m.get("tool_calls") or ():
                names[call["id"]] = call["function"]["name"]
        elif m.get("role") == "tool":
            out.append((names.get(m.get("tool_call_id"), "?"), _data(m.get("content"))))
    return out


class ScreeningStub:
    """A Flowgentic transport that answers each call with the workflow's next tool call."""

    def __init__(self, bench: Benchmark, latency_s: float = 0.0, tool_retries: int = 3) -> None:
        self.bench = bench
        self.latency_s = latency_s
        self.tool_retries = tool_retries
        self.model = "screening-stub"

    def next_call(self, messages: list[dict]) -> tuple[str, dict] | None:
        bench = self.bench
        human = next(m for m in messages if m.get("role") == "user")
        cid = CID.search(str(human.get("content"))).group(0)
        results = tool_results(messages)

        def by(name: str) -> list[dict]:
            return [d for n, d in results if n == name]

        lookups, sims, analyses = by("lookup_reference"), by("simulate"), by("analyze")
        if by("report"):
            return None
        found = [d for d in lookups if "error" not in d]
        if len(found) < bench.lookup_calls:
            failed = 0
            for d in reversed(lookups):
                if "error" not in d:
                    break
                failed += 1
            if failed > self.tool_retries:
                raise ToolGaveUp(
                    f"lookup_reference failed {failed} times in a row: {lookups[-1]['error']}"[:400]
                )
            return "lookup_reference", {"material": cid}
        start = physics.reference(physics.candidate(physics.index_of(cid), bench), bench)
        if not results:
            return "simulate", {"temperature": start[0], "doping": start[1]}
        name, data = results[-1]
        if name == "lookup_reference":
            return "simulate", {
                "temperature": data["synthesis_temperature_k"],
                "doping": data["dopant_fraction"],
            }
        if name == "simulate" and "result_key" in data:
            return "analyze", {"result_key": data["result_key"]}
        if (
            name == "analyze"
            and "next" in data
            and not data.get("converged")
            and len(sims) < bench.max_sims
        ):
            return "simulate", {
                "temperature": data["next"]["temperature"],
                "doping": data["next"]["doping"],
            }
        good = [a for a in analyses if "score" in a]
        if not good:
            return "report", {"temperature": start[0], "doping": start[1], "score": 0.0}
        best = max(good, key=lambda a: a["score"])
        return "report", {
            "temperature": best["temperature"],
            "doping": best["doping"],
            "score": best["score"],
        }

    async def complete(self, messages, tools=None, **kwargs) -> Reply:
        await asyncio.sleep(self.latency_s)
        call = self.next_call(list(messages))
        n = sum(1 for m in messages if m.get("role") == "tool")
        if call is None:
            content, calls = "DONE", ()
        else:
            content, calls = "", (ToolCall(id=f"call-{n}", name=call[0], args=call[1]),)
        prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        out_chars = len(content) + sum(len(json.dumps(c.args)) for c in calls)
        return Reply(
            content=content,
            tool_calls=calls,
            usage=Usage(prompt_tokens=prompt_chars // 4, completion_tokens=out_chars // 4),
            finish_reason="tool_calls" if calls else "stop",
            model=self.model,
        )
