"""The benchmark's parameters (its knobs).

The workflow is one setting of an application-independent benchmark: the generator takes these
parameters, and materials screening is the default. Every run records the parameters it used.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

# wfMiniAPI's compute kernels that take only (device, size): the ones a simulation can run
SIM_KERNELS = ("matMulSimple2D", "axpy", "axpy_fast", "generateRandomNumber")


@dataclass(frozen=True)
class Benchmark:
    """What each agent does, and how much work that is: the benchmark's parameters.

    Materials screening is the default setting. Every run records these in its summary.

    Tool mix and calls. An agent following the workflow calls `lookup_reference` (MCP)
    `lookup_calls` times, then `simulate` and `analyze` once per simulation, then `report`; each
    tool call is one model call, plus the final answer. So model calls per agent are
    `lookup_calls + 2 * simulations + 2` (`model_calls_per_agent`).

    Simulations per agent. With `sims_per_agent` 0 the physics decides: an agent starts from its
    reference conditions, at a seeded distance of `start_offset` widths from its candidate's
    optimum in each parameter; each analysis points at a next point `contraction` times as far from
    the optimum; the agent has converged when its score is within `converge_tol` of its
    candidate's best. With the defaults that is 2 or 3 simulations. With `sims_per_agent` k (1 to
    `max_sims`), every agent needs exactly k: the analyses say converged only at the k-th, and the
    one before it points at the optimum. `verify_tol` is how close a report must come to the best.

    Simulation durations, drawn per simulation from a seed of the candidate and the parameters,
    so a repeat asks for the same work: `lognormal` (median `sim_median_s`, shape `sim_sigma`),
    `fixed` (`sim_median_s`) or `uniform` (between the two values of `sim_range_s`).
    `sim_out_elems` float64 values go to the store per simulation.

    Simulation compute. A simulation spins a CPU core for its duration (`sim_burn`). With
    `sim_kernel`, it runs that wfMiniAPI compute kernel instead, sized `sim_kernel_size`, on
    `sim_kernel_device`, over and over for the same duration (at least once): the benchmark still
    sets how long a simulation takes, and the kernel gives it a real footprint (floating-point
    work, memory, a GPU). Off by default; the result written to the store is the same either way.

    Suspension. Before each simulation an agent suspends (checkpoint to the store, state dropped,
    resumed from the store when the simulation ends) with probability `suspend_fraction`, drawn per
    simulation from a seed; otherwise it waits for the simulation in memory.

    Simulation cost per candidate. Each candidate gets a seeded cost scale s (lognormal, median 1,
    shape `sim_cost_sigma`) that multiplies its simulations' durations. With 0, s is 1 for all.

    The reasoning knob. A decision call is a model call that follows a reference lookup or an
    analysis (or, without lookups, the first call): it chooses the next simulation or stops. Each
    agent's decision calls get a thinking budget in tokens: `think_budget` B for every agent
    (`think_policy` "fixed"), or B times its candidate's cost scale s, clipped to
    `think_budget_max` ("proportional"). The other calls keep thinking off. Reasoning costs the
    thinking tokens, and buys a modeled gain: the analysis of a simulation whose conditions a call
    with budget b chose points at next conditions `contraction / (1 + think_gain * log2(1 + b /
    256))` times as far from the optimum. b is min(budget, thinking tokens the call used) with
    `think_gain_on` "used" (the default), or the budget asked for with "asked". With "used" the
    simulations an agent needs depend on the model's thinking, so they are not known in advance:
    the run verifies each agent's report, not its count of simulations. With `think_budget` 0 the
    workflow is the one without the knob; with `think_gain` 0, reasoning costs tokens and changes
    nothing else.

    Model calls not tied to HPC work. With `plan_calls` N, each decision point (the choice of each
    simulation, and the choice to report) starts with N calls to a cheap in-process tool, `note`:
    the agent records its plan before its first simulation, and a reflection after each analysis.
    An agent that runs s simulations has s + 1 decision points, so model calls per agent are
    `lookup_calls + 2 * s + 2 + N * (s + 1)`. The physics is unchanged.

    Longer contexts. With `tool_result_tokens` K, each reference lookup returns, beside its fields,
    a reference document of about K tokens (seeded text), and each analysis its spectrum, sampled to
    about K tokens of numbers. The fields an agent needs stay as they are, so a correct run does not
    depend on reading the padding. With 0 the results are the ones before this knob.
    """

    agents: int = 100
    seed: int = 0
    # tools and their calls
    max_sims: int = 4
    sims_per_agent: int = 0  # 0: the physics decides (2-3); k: exactly k per agent
    lookup_calls: int = 1  # MCP reference look-ups per agent
    # simulations
    sim_distribution: str = "lognormal"
    sim_median_s: float = 10.0
    sim_sigma: float = 0.5
    sim_range_s: tuple[float, float] = (30.0, 300.0)  # the uniform distribution's bounds
    sim_out_elems: int = 1024
    sim_burn: bool = True  # spin a CPU for the duration; False sleeps (local runs, tests)
    # a wfMiniAPI compute kernel run for each simulation's duration, in place of the spin ("": off)
    sim_kernel: str = ""
    sim_kernel_size: int = 256  # matMulSimple2D: size x size matrices; the others: size values
    sim_kernel_device: str = "cpu"  # cpu (NumPy) or gpu (CuPy)
    # the MCP tool
    mcp_latency_ms: tuple[float, float] = (5.0, 50.0)
    # suspension: the share of simulations an agent suspends before
    suspend_fraction: float = 1.0
    # convergence of the synthetic physics
    start_offset: tuple[float, float] = (0.3, 1.2)
    contraction: float = 0.2
    converge_tol: float = 0.03
    verify_tol: float = 0.05
    # agents per process
    agents_per_process: int = 5000
    # simulation cost per candidate: the shape of its lognormal cost scale (median 1)
    sim_cost_sigma: float = 0.0
    # the reasoning knob: thinking budget per decision call (tokens), how it is set per agent, its gain
    think_budget: int = 0
    think_policy: str = "fixed"  # fixed: B for every agent; proportional: B times the cost scale
    think_budget_max: int = 8192  # the clip of a proportional budget
    think_gain: float = 0.0
    think_gain_on: str = "used"  # used: min(budget, thinking tokens used); asked: the budget
    # model calls not tied to HPC work: calls to the `note` tool per decision point
    plan_calls: int = 0
    # longer contexts: tokens of padding in each reference lookup's and analysis's result (0: none)
    tool_result_tokens: int = 0

    def __post_init__(self) -> None:
        refused = [why for wrong, why in self._checks() if wrong]
        if refused:
            raise ValueError("invalid benchmark: " + "; ".join(refused))

    def _checks(self):
        """(wrong, why) for each knob. They are independent, so all are checked and a refusal
        names every invalid value at once."""
        low, high = self.sim_range_s
        yield self.max_sims < 1, f"max_sims must be at least 1 (got {self.max_sims})"
        yield (
            not 0 <= self.sims_per_agent <= self.max_sims,
            "sims_per_agent must be 0 (the physics decides) or 1 to max_sims "
            f"(got {self.sims_per_agent})",
        )
        yield self.lookup_calls < 0, f"lookup_calls must be 0 or more (got {self.lookup_calls})"
        yield (
            self.sim_distribution not in ("lognormal", "fixed", "uniform"),
            f"unknown sim_distribution {self.sim_distribution!r}",
        )
        yield (
            low > high or low < 0,
            f"sim_range_s must be (low, high) with 0 <= low <= high (got {self.sim_range_s})",
        )
        yield (
            not 0.0 <= self.suspend_fraction <= 1.0,
            f"suspend_fraction must be in [0, 1] (got {self.suspend_fraction})",
        )
        yield (
            not 0 < self.contraction < 1,
            f"contraction must be in (0, 1) (got {self.contraction})",
        )
        yield (
            self.sim_cost_sigma < 0,
            f"sim_cost_sigma must be 0 or more (got {self.sim_cost_sigma})",
        )
        yield (
            self.think_budget < 0 or self.think_budget_max < 0,
            "think_budget and think_budget_max must be 0 or more "
            f"(got {self.think_budget}, {self.think_budget_max})",
        )
        yield (
            self.think_policy not in ("fixed", "proportional"),
            f"unknown think_policy {self.think_policy!r}",
        )
        yield self.think_gain < 0, f"think_gain must be 0 or more (got {self.think_gain})"
        yield (
            self.think_gain_on not in ("used", "asked"),
            f"unknown think_gain_on {self.think_gain_on!r}",
        )
        yield self.plan_calls < 0, f"plan_calls must be 0 or more (got {self.plan_calls})"
        yield (
            self.tool_result_tokens < 0,
            f"tool_result_tokens must be 0 or more (got {self.tool_result_tokens})",
        )
        if self.sim_kernel:
            yield (
                self.sim_kernel not in SIM_KERNELS,
                f"unknown sim_kernel {self.sim_kernel!r}: one of {', '.join(SIM_KERNELS)}",
            )
            yield (
                self.sim_kernel_size < 1,
                f"sim_kernel_size must be at least 1 (got {self.sim_kernel_size})",
            )
            yield (
                self.sim_kernel_device not in ("cpu", "gpu"),
                f"sim_kernel_device must be cpu or gpu (got {self.sim_kernel_device!r})",
            )

    @property
    def sim_mean_s(self) -> float:
        cost = math.exp(self.sim_cost_sigma**2 / 2)  # the mean of the candidates' cost scales
        if self.sim_distribution == "fixed":
            return self.sim_median_s * cost
        if self.sim_distribution == "uniform":
            return sum(self.sim_range_s) / 2 * cost
        return self.sim_median_s * math.exp(self.sim_sigma**2 / 2) * cost

    def model_calls_per_agent(self, sims: int) -> int:
        """Model calls of an agent that follows the workflow and runs `sims` simulations."""
        return self.lookup_calls + 2 * sims + 2 + self.plan_calls * self.decisions(sims)

    @staticmethod
    def decisions(sims: int) -> int:
        """Decision points of an agent that runs `sims` simulations: one before each simulation, and
        the choice to report."""
        return sims + 1

    def asdict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Benchmark:
        names = {f.name for f in dataclasses.fields(cls)}
        clean = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
        return cls(**clean)
