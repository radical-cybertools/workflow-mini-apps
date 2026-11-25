"""The candidate set and its seeded synthetic physics.

Each candidate material has a hidden optimum: synthesis temperature `t_opt` (K) and dopant
fraction `d_opt`, where its figure of merit reaches `best`. The score falls off as a Gaussian in
both parameters:

    score(T, d) = best * exp(-((T - t_opt) / T_WIDTH)**2 - ((d - d_opt) / D_WIDTH)**2)

so the best reachable score of every candidate is known in advance, and a run can be verified.

The reference-data server tells an agent its candidate's reference conditions: the optimum moved by
a seeded offset of `start_offset` widths in each parameter. Each analysis of a simulation points at
the next conditions, `contraction` times as far from the optimum as the ones simulated, and says
whether the score has converged. Everything is a function of the run's seed and the candidate, so
every process computes the same values.

The reasoning knob. Each candidate has a seeded simulation-cost scale (`cost_scale`), which
multiplies its simulations' durations and sets its agent's thinking budget under the proportional
policy (`think_budget`). Reasoning makes the analyses point closer to the optimum: the analysis of
a simulation whose conditions a decision call with budget b chose uses `contraction(b)`, where b
is min(budget, thinking tokens used) by default (`gain_budget`). Every agent's decision calls get
the same budget, so with the gain on the budget asked for `expected_sims` knows each agent's count;
on the tokens used it gives a lower bound, and the run verifies the reports.

Longer contexts. With `tool_result_tokens` K, the reference data carries a document of about K
tokens (`reference_document`): seeded sentences of words that are one token each with Qwen3's
tokenizer (and one more per sentence, its full stop), so the count holds to within 1 % at K of
500 and more. An analysis carries its spectrum sampled to K / `TOKENS_PER_VALUE` values
(`spectrum_sample`): each value has four decimals, about 7.9 tokens with its separator.

Standard library only: the simulation function imports this module in RHAPSODY's workers.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .benchmark import Benchmark

T_RANGE = (300.0, 1500.0)  # K
D_RANGE = (0.0, 0.25)  # dopant fraction
T_WIDTH = 250.0
D_WIDTH = 0.05
TARGETS = (
    "thermoelectric figure of merit",
    "ionic conductivity",
    "photocatalytic yield",
    "band-gap match",
)
SPECTRUM_POINTS = 64
TOKENS_PER_VALUE = (
    8  # a spectrum value in JSON (0.xxxx and its separator): 7.9 tokens with Qwen3's tokenizer
)

# Words that are one token each, after a space, with Qwen3's tokenizer; the sentences start with CAPS
CAPS = tuple("The This These We In At For Our Each Both Such Its As When After Before".split())
WORDS = """sample samples phase phases crystal lattice structure surface grain grains boundary boundaries oxide
film films layer layers doping temperature pressure heat energy charge carrier carriers transport
conductivity mobility thermal electrical optical magnetic chemical stability stable data measured
measurement measurements reported report observed value values range peak peaks shift band gap
density volume mass ratio rate rates order disorder defect defects vacancy site sites atom atoms ion
ions bond bonds metal metals alloy alloys compound compounds material materials method methods
process processing synthesis synthesized prepared cooling heating growth formed formation increase
increased decrease decreased higher lower above below between within during after before under over
across along near strong weak large small high low similar different related previous recent known
expected typical model models theory calculated calculation result results analysis study studies
work group effect effects response performance property properties composition concentration content
fraction level levels point points condition conditions time times step steps limit series set sets
case cases figure table section shows show shown suggests suggest indicates indicate consistent
agreement better best close closely slightly significantly mainly mostly also however therefore
while which where when that these those this with from into than more most less both each other such
only same first second third final initial main major minor further found used using based reference
reaction reactions precursor powder solution solutions substrate electrode electrodes cell cells
device devices efficiency yield yields catalyst activity active area pore pores spectrum spectra
signal signals intensity width center region regions the of and to in is was are were be by on as at
an it its for not can has have had been may""".split()


@dataclass(frozen=True)
class Candidate:
    index: int
    cid: str
    target: str
    t_opt: float
    d_opt: float
    best: float
    t_ref: float
    d_ref: float


def cid_of(index: int) -> str:
    return f"mat-{index:06d}"


def index_of(cid: str) -> int:
    return int(cid.split("-", 1)[1])


def _clip(x: float, lo_hi: tuple[float, float]) -> float:
    return min(max(x, lo_hi[0]), lo_hi[1])


def candidate(index: int, bench: Benchmark) -> Candidate:
    """Candidate `index` of the run with `bench`'s seed. `random.Random(str)` is stable across processes."""
    # the seeds' strings name the benchmark's first implementation: kept, so results stay the same
    r = random.Random(f"rascal/{bench.seed}/{index}")
    t_opt = r.uniform(600.0, 1200.0)
    d_opt = r.uniform(0.05, 0.20)
    best = r.uniform(0.60, 0.98)
    lo, hi = bench.start_offset
    t_ref = _clip(t_opt + r.choice((-1, 1)) * r.uniform(lo, hi) * T_WIDTH, T_RANGE)
    d_ref = _clip(d_opt + r.choice((-1, 1)) * r.uniform(lo, hi) * D_WIDTH, D_RANGE)
    return Candidate(index, cid_of(index), r.choice(TARGETS), t_opt, d_opt, best, t_ref, d_ref)


def score(c: Candidate, temperature: float, doping: float) -> float:
    return c.best * math.exp(
        -(((temperature - c.t_opt) / T_WIDTH) ** 2) - ((doping - c.d_opt) / D_WIDTH) ** 2
    )


def converged(c: Candidate, s: float, bench: Benchmark) -> bool:
    return s >= (1.0 - bench.converge_tol) * c.best


def cost_scale(index: int, bench: Benchmark) -> float:
    """Candidate `index`'s simulation-cost scale: lognormal, median 1, shape `sim_cost_sigma` (1 with 0)."""
    if bench.sim_cost_sigma <= 0:
        return 1.0
    return random.Random(f"rascal-cost/{bench.seed}/{index}").lognormvariate(
        0.0, bench.sim_cost_sigma
    )


def think_budget(index: int, bench: Benchmark) -> int:
    """The thinking budget (tokens) of candidate `index`'s agent's decision calls, by `think_policy`."""
    b = bench.think_budget
    if b <= 0:
        return 0
    if bench.think_policy == "proportional":
        return min(round(b * cost_scale(index, bench)), bench.think_budget_max)
    return b


def gain_budget(asked: int, used: int | None, bench: Benchmark) -> int:
    """The budget the gain counts for a decision call: min(asked, thinking tokens used) with
    `think_gain_on` "used" (when the count is known), the budget asked for with "asked"."""
    if bench.think_gain_on == "used" and used is not None:
        return min(asked, used)
    return asked


def counts_known(bench: Benchmark) -> bool:
    """Whether `expected_sims` knows each agent's simulations in advance: not when the gain depends
    on the thinking tokens the model used."""
    return not (bench.think_budget > 0 and bench.think_gain > 0 and bench.think_gain_on == "used")


def contraction(budget: int, bench: Benchmark) -> float:
    """The analyses' contraction after a decision call with this budget: `contraction` with no budget
    or no gain, smaller with more budget, with diminishing returns."""
    if budget <= 0 or bench.think_gain <= 0:
        return bench.contraction
    return bench.contraction / (1.0 + bench.think_gain * math.log2(1.0 + budget / 256.0))


def next_point(
    c: Candidate, temperature: float, doping: float, bench: Benchmark, budget: int = 0
) -> tuple[float, float]:
    """Where the analysis points after a simulation at (temperature, doping), rounded as it reports it.
    `budget` is the thinking budget of the decision call that chose (temperature, doping)."""
    k = contraction(budget, bench)
    t = _clip(c.t_opt + k * (temperature - c.t_opt), T_RANGE)
    d = _clip(c.d_opt + k * (doping - c.d_opt), D_RANGE)
    return round(t, 1), round(d, 4)


def optimum(c: Candidate) -> tuple[float, float]:
    return round(c.t_opt, 1), round(c.d_opt, 4)


def analysis(
    c: Candidate, temperature: float, doping: float, n: int, bench: Benchmark, budget: int = 0
) -> tuple[float, bool, tuple | None]:
    """The analysis of the agent's `n`-th simulation (from 1): `(score, converged, next point or None)`.

    With `bench.sims_per_agent` 0, an agent has converged once its score is within `converge_tol` of
    the best, and the next point is `contraction(budget)` times as far from the optimum, where
    `budget` is the thinking budget of the decision call that chose the simulation's conditions.
    With `sims_per_agent` k, it has converged at its k-th simulation, and the (k-1)-th analysis
    points at the optimum.
    """
    s = score(c, temperature, doping)
    k = bench.sims_per_agent
    if k:
        done = n >= k
        nxt = (
            None
            if done
            else (optimum(c) if n == k - 1 else next_point(c, temperature, doping, bench, budget))
        )
    else:
        done = converged(c, s, bench)
        nxt = None if done else next_point(c, temperature, doping, bench, budget)
    return s, done, nxt


def reference(c: Candidate, bench: Benchmark) -> tuple[float, float]:
    """The reference conditions an agent starts from: the optimum when one simulation must do."""
    if bench.sims_per_agent == 1:
        return optimum(c)
    return round(c.t_ref, 1), round(c.d_ref, 4)


def expected_sims(c: Candidate, bench: Benchmark, budget: int | None = None) -> int:
    """Simulations an agent that follows every analysis runs: from the reference until it converges.
    Its decision calls all have its budget, so every analysis uses `contraction(think_budget)`, or
    `contraction(budget)` when given. With the gain on the tokens used (`counts_known` false), this
    is the count if every call used its whole budget: a lower bound."""
    t, d = reference(c, bench)
    b = think_budget(c.index, bench) if budget is None else budget
    for n in range(1, bench.max_sims + 1):
        _, done, nxt = analysis(c, t, d, n, bench, b)
        if done:
            return n
        t, d = nxt
    return bench.max_sims


def suspends(cid: str, n: int, bench: Benchmark) -> bool:
    """Whether the agent suspends before its `n`-th simulation: a seeded draw of `suspend_fraction`."""
    f = bench.suspend_fraction
    if f >= 1.0:
        return True
    if f <= 0.0:
        return False
    return random.Random(f"rascal-suspend/{bench.seed}/{cid}/{n}").random() < f


def verifies(c: Candidate, temperature: float, doping: float, bench: Benchmark) -> bool:
    return score(c, temperature, doping) >= (1.0 - bench.verify_tol) * c.best


def sim_seconds(c: Candidate, temperature: float, doping: float, bench: Benchmark) -> float:
    """The simulation's duration, drawn from a seed of the candidate and the parameters, times the
    candidate's cost scale."""
    if bench.sim_distribution == "fixed":
        seconds = bench.sim_median_s
    else:
        r = random.Random(f"rascal-sim/{bench.seed}/{c.cid}/{temperature:.3f}/{doping:.5f}")
        if bench.sim_distribution == "uniform":
            seconds = r.uniform(*bench.sim_range_s)
        elif bench.sim_median_s > 0:
            seconds = r.lognormvariate(math.log(bench.sim_median_s), bench.sim_sigma)
        else:
            seconds = 0.0
    if bench.sim_cost_sigma > 0:
        seconds *= cost_scale(c.index, bench)
    return seconds


def spectrum(c: Candidate, temperature: float, doping: float, n: int) -> list[float]:
    """The simulated property curve: `n` values whose peak is the score.

    Layout of the result tensor: `[temperature, doping, y_0, ..., y_{n-1}]`.
    """
    s = score(c, temperature, doping)
    centre = (n - 1) * (temperature - T_RANGE[0]) / (T_RANGE[1] - T_RANGE[0])
    width = max(n / 16, 1.0)
    ys = [s * math.exp(-(((i - centre) / width) ** 2)) for i in range(n)]
    ys[min(max(round(centre), 0), n - 1)] = s  # the peak is exactly the score
    return [temperature, doping, *ys]


def lookup(c: Candidate, bench: Benchmark) -> dict:
    """What the reference-data server knows about a candidate: a related compound's conditions, and
    with `tool_result_tokens` a reference document of about that many tokens."""
    t, d = reference(c, bench)
    out = {
        "material": c.cid,
        "reference_compound": f"{c.cid}-ref",
        "target_property": c.target,
        "synthesis_temperature_k": t,
        "dopant_fraction": d,
        "figure_of_merit": round(score(c, t, d), 4),
    }
    if bench.tool_result_tokens > 0:
        out["reference_document"] = reference_document(c, bench)
    return out


def reference_document(c: Candidate, bench: Benchmark, tokens: int | None = None) -> str:
    """Seeded text of about `tokens` (default `bench.tool_result_tokens`) tokens: sentences of 8 to
    18 one-token words, each one more token for its full stop."""
    k = bench.tool_result_tokens if tokens is None else tokens
    r = random.Random(f"rascal-doc/{bench.seed}/{c.cid}")
    sentences, t = [], 0
    while t < k:
        m = min(r.randint(8, 18), max(k - t - 1, 1))
        sentences.append(" ".join([r.choice(CAPS)] + [r.choice(WORDS) for _ in range(m - 1)]) + ".")
        t += m + 1
    return " ".join(sentences)


def spectrum_sample(values, tokens: int, key: str) -> list[float]:
    """A simulation's spectrum, sampled to about `tokens` tokens of JSON numbers: tokens /
    `TOKENS_PER_VALUE` values at evenly spaced points of `values`, each with a seeded noise floor
    (0.001 to 0.01, as a measured spectrum has) and four decimals. `key` seeds the noise."""
    m = len(values)
    n = max(1, round(tokens / TOKENS_PER_VALUE))
    if m == 0:
        return []
    r = random.Random(f"rascal-spectrum/{key}")
    step = (m - 1) / (n - 1) if n > 1 else 0.0
    return [round(float(values[round(i * step)]) + r.uniform(0.001, 0.01), 4) for i in range(n)]


def mcp_latency_ms(cid: str, call: int, bench: Benchmark) -> float:
    lo, hi = bench.mcp_latency_ms
    return random.Random(f"rascal-mcp/{bench.seed}/{cid}/{call}").uniform(lo, hi)
