"""The agent's instructions.

With `bench.plan_calls` N the prompt adds the `note` tool and N notes at each decision point: a plan
before the first simulation, a reflection after each analysis. Without it the prompt is the one
before that knob, word for word."""

from .benchmark import Benchmark
from .physics import D_RANGE, T_RANGE, Candidate, reference

SYSTEM = """You are a materials-screening agent. You optimize the synthesis conditions of one candidate \
material: the temperature in kelvin ({t_lo:.0f} to {t_hi:.0f}) and the dopant fraction ({d_lo} to {d_hi}). \
The goal is the highest score for the target property.

Tools:
- lookup_reference(material): reference data for the material, with suggested starting conditions.
- simulate(temperature, doping): runs a simulation and returns a result_key.
- analyze(result_key): scores a simulation's result, says whether the score has converged, and \
suggests the next conditions.
- report(temperature, doping, score): records your final answer.{note_tool}

Follow these steps exactly, calling one tool at a time:
{steps}"""
NOTE_TOOL = """
- note(text): records a note in your notebook: your plan, or your reflection on a result."""

STEPS = [
    "Call analyze with the result_key that simulate returned.",
    'If analyze says converged is false, call simulate with the temperature and doping in its "next" field, '
    "then analyze again. You may run at most {max_sims} simulations in all.",
    "When analyze says converged is true, or after {max_sims} simulations, call report with the temperature, "
    "doping and score of the best analysis so far.",
    "After report returns, reply with the single word DONE.",
]
START = {
    0: ["Call simulate with the starting conditions given in the task."],
    1: [
        "Call lookup_reference with the material's name.",
        "Call simulate with the reference's synthesis_temperature_k and dopant_fraction.",
    ],
}
START_MANY = [
    "Call lookup_reference with the material's name, {n} times in all, one call at a time.",
    "Call simulate with the last reference's synthesis_temperature_k and dopant_fraction.",
]

TASK = "Candidate material: {cid}. Target property: {target}. Start now."
TASK_WITH_START = (
    "Candidate material: {cid}. Target property: {target}. "
    "Starting conditions: temperature {t} K, dopant fraction {d}. Start now."
)

# With notes (`bench.plan_calls`): N notes before the first simulation, and after each analysis
NOTE_PLAN = (
    "Call note {times} to record your plan: which conditions you will simulate first, and why."
)
NOTE_REFLECT = (
    "After every analysis, the last one too, call note {times} to record your reflection on the analysis "
    "and your plan for the next step."
)
STEPS_NOTES = [
    "Call analyze with the result_key that simulate returned.",
    NOTE_REFLECT,
    'If the analysis said converged is false, call simulate with the temperature and doping in its "next" field, '
    "then analyze and note again. You may run at most {max_sims} simulations in all.",
    "When an analysis says converged is true, or after {max_sims} simulations, write its notes, then call report "
    "with the temperature, doping and score of the best analysis so far.",
    "After report returns, reply with the single word DONE.",
]

NUDGE = "You have not called report yet. Call report now with the temperature, doping and score of your best analysis."


def times(n: int) -> str:
    return "once" if n == 1 else ("twice" if n == 2 else f"{n} times") + ", one call at a time,"


def system_prompt(bench: Benchmark) -> str:
    """The workflow's steps, for `bench.lookup_calls` MCP look-ups, and
    `bench.plan_calls` notes per decision point."""
    n = bench.lookup_calls
    start = START.get(n) or [s.format(n=n) for s in START_MANY]
    steps, note_tool = STEPS, ""
    if bench.plan_calls:
        start = start[:-1] + [NOTE_PLAN.format(times=times(bench.plan_calls))] + start[-1:]
        steps, note_tool = (
            [s.replace("{times}", times(bench.plan_calls)) for s in STEPS_NOTES],
            NOTE_TOOL,
        )
    steps = "\n".join(
        f"{i}. {s.format(max_sims=bench.max_sims)}" for i, s in enumerate(start + steps, 1)
    )
    return SYSTEM.format(
        t_lo=T_RANGE[0],
        t_hi=T_RANGE[1],
        d_lo=D_RANGE[0],
        d_hi=D_RANGE[1],
        steps=steps,
        note_tool=note_tool,
    )


def task_message(c: Candidate, bench: Benchmark) -> str:
    if bench.lookup_calls == 0:
        t, d = reference(c, bench)
        return TASK_WITH_START.format(cid=c.cid, target=c.target, t=t, d=d)
    return TASK.format(cid=c.cid, target=c.target)
