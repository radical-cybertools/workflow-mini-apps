# Agentic screening: a mini-app of agents and many engines

A synthetic materials-screening campaign. One agent per candidate material. It:
- looks the material up, an MCP tool;
- runs simulations, each an AsyncFlow task;
- analyzes each result;
- reports the best synthesis conditions it found.

The physics is seeded, so every candidate's best is known and the run verifies itself.

The agents run in populations. Each population is a node of the mini-app's AsyncFlow graph, with
an AsyncFlow engine of its own for its simulations. The run's `MiniApp` assembles and verifies the
whole run:
- each population's engine is a part of the run;
- the agents' OpenTelemetry export and the application's rows are components with verifiers;
- checks run across them.

So this example uses `MiniApp` for runs of many engines and components.

| Path | What it holds |
|---|---|
| `miniapp_screening.py` | The mini-app: the engine, its telemetry, `MiniApp`, the workflow and the script's shutdown |
| `screening/benchmark.py` | The benchmark's knobs (`Benchmark`): agents, simulations per agent, their duration, lookups, suspension, the seed |
| `screening/physics.py`, `prompts.py`, `verify.py` | The seeded workload: candidates and their best, the agent's instructions, the verification and ranking |
| `screening/sim.py`, `tools.py` | The simulation a task runs, its arguments, and the budget of simulations per agent |
| `screening/mcp_server.py` | The reference-data MCP server, one tool, `lookup_reference` |
| `screening/stub.py` | A model for runs without GPUs: it plays the workflow, so a run's per-agent results are known |
| `screening/agent.py` | The agent definition the AUTONMI framework runs |
| `jobs/` | The scale ladder: the campaign at 1,000 to 300,000 agents, as a chain of Slurm jobs ([jobs/README.md](jobs/README.md)) |

The physics, the prompts, the benchmark and the verification need nothing outside the package.

## What it needs

- **The AUTONMI framework.** It runs populations of agents as nodes of an AsyncFlow graph, on Flowgentic's agent loop. It is installed separately: the RADICAL team gives access to it. It brings Flowgentic, AsyncFlow, RHAPSODY and the OpenTelemetry SDK.
- **This repository's `wfMiniAPI`,** with `MiniApp` for runs of many engines and components.
- **Under Dragon:** Dragon and RaDEX, as RHAPSODY's native backend uses them.

Without the framework, this repository's tests skip this example, apart from the benchmark's own test, the check of the pinned commit, and the contract check, which reads its source.

**The framework's commit.** This example is tested on AUTONMI commit `cdda559fcd78dbf42671305c3fdb0b6fa97d451b` (`main` of 26 Nov 2026, with its public API). That is `AUTONMI_COMMIT` in `screening/pin.py`, and a test keeps this line and the constant equal. Install it with the versions AUTONMI's own `requirements.txt` pins, then this repository without its dependencies, so that AUTONMI's pins hold. From a clone of each, side by side, with Python 3.12 and [uv](https://docs.astral.sh/uv/):

```shell
git -C autonmi checkout cdda559fcd78dbf42671305c3fdb0b6fa97d451b
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r autonmi/requirements.txt pyyaml pytest
uv pip install --no-deps -e autonmi -e workflow-mini-apps
```

AUTONMI's `requirements.txt` pins RHAPSODY at `eac7ceb`, while this repository's `workflow` extra pins `142f396`. The example is tested with AUTONMI's RHAPSODY, which is why the last line skips the extras. Across nodes under Dragon, the populations also need RHAPSODY's native backend with `hosts=` (radical-cybertools/rhapsody#169).

A run checks the commit it imports and records it in `<run-dir>/autonmi.json`. The commit comes from git for an editable clone, from a `.git-commit` file at the root of a copy made without git (`git archive <commit> | tar -x -C DIR; echo <commit> > DIR/.git-commit`), or from the installer's record for a package installed from git. A run at another commit, at an unknown one, or with changed files under the package prints a warning on stderr and goes on.

## Running it

Locally, with the stub model: threads and a thread pool, no Dragon, no GPU.

```shell
python examples/agentic-screening/miniapp_screening.py --agents 200 --populations 2 \
    --sim-median 0.05 --no-burn
```

Under Dragon, with a model job's vLLM servers. The job writes `endpoints` (one base URL per server) and `api-key`.

```shell
python -m dragon -s examples/agentic-screening/miniapp_screening.py --runtime dragon \
    --populations 2 --workers 52 --model openai \
    --endpoints-file FLEET/endpoints --api-key-file FLEET/api-key
```

The script puts `screening/` first on `PYTHONPATH`, so that the processes the run starts import it too: the MCP servers and, under Dragon, the populations and their workers.

`--help` lists every knob with its default, in groups:
- **the campaign:** `--agents`, `--seed`, `--populations`, `--runtime`, `--timeout`;
- **what the agents do:**
  - simulation length (`--sim-median`, `--sim-sigma`, `--sim-distribution`, `--sim-range`);
  - the cost spread (`--sim-cost-sigma`), simulations per agent (`--sims-per-agent`, `--max-sims`), burn or sleep (`--no-burn`), a compute kernel (`--sim-kernel`, `--sim-kernel-size`, `--sim-kernel-device`);
  - simulation output (`--sim-out-elems`), reference lookups (`--lookup-calls`), MCP latency (`--mcp-latency-ms`);
  - planning calls (`--plan-calls`), long tool results (`--tool-result-tokens`), suspension (`--suspend-fraction`, `--no-suspend`);
  - the thinking knobs (`--think-budget`, `--think-policy`, `--think-budget-max`, `--think-gain`, `--think-gain-on`);
- **the model:** `--model`, the stub's `--stub-latency`, `--stub-replicas`, `--stub-kv-tokens`, `--stub-decode-rate`, `--stub-think-median` and `--stub-think-sigma`, the servers' `--endpoints-file`, `--model-name`, `--max-tokens` and `--tokenizer`;
- **admission:** the request cap (`--cap`), the token cap (`--cap-tokens`, `--reply-reserve`), `--admission`;
- **the system's processes:** `--workers`, the staggered start (`--start-batch`, `--start-interval`), `--mcp-servers`, `--toolservices`, `--mcp-clients`, `--mcp-per-client`, and RaDEX's `--managers` and `--manager-gib`;
- **telemetry:** `--span-ratio` (1 keeps every span), `--metrics-interval`.

**The reasoning knob.** A decision call follows a reference lookup or an analysis (with planning calls, after its N-th note), and chooses the next simulation or the report.
- **The budget.** With `--think-budget` B, each decision call may think for B tokens; with `--think-policy proportional`, for B times the candidate's cost scale, clipped at `--think-budget-max`. The other calls keep thinking off.
- **The gain.** With `--think-gain` above 0, the analysis of a simulation points closer to the optimum the more its decision call thought, so the agent needs fewer simulations. `--think-gain-on used` counts the thinking tokens used, when fewer than the budget; `asked` counts the budget.
- **With the stub,** a decision call thinks for min(budget, a seeded natural length): median `--stub-think-median`, shape `--stub-think-sigma`, at `--stub-decode-rate` tokens per second.
- **With vLLM,** a decision call turns the model's thinking on (Qwen3's `enable_thinking`) and sends `thinking_token_budget`, with room for it in `max_tokens`. The servers need a reasoning parser and a thinking budget enabled. vLLM's chat API doesn't count thinking tokens apart, so they are counted from the reasoning text the parser splits out: with the model's tokenizer when `--tokenizer` gives it, otherwise as the reasoning's share of the reply's characters, times its completion tokens.
- **The nudge.** A model sometimes answers in text where it should call `report`, more often when its thinking was cut at a small budget. Such an agent is told once to report, as the benchmark's agents are; the row counts its `nudges`, and the check of the model calls counts each nudge as one call more than the script's.

**A compute kernel in each simulation.** By default a simulation spins a CPU core for its drawn duration. With `--sim-kernel`, it runs one of wfMiniAPI's compute kernels instead, over and over for the same duration (at least once):
- **The kernels:** `matMulSimple2D` (size × size float32 matrices), `axpy`, `axpy_fast` and `generateRandomNumber` (size values).
- **The size:** `--sim-kernel-size`, 256 by default.
- **The device:** `--sim-kernel-device` is `cpu` (NumPy) or `gpu` (CuPy, the `gpu` extra).

The benchmark still sets how long each simulation takes, so the agents, the checks and the results are the same as without the kernel. The kernel gives the simulation a real footprint: floating-point work, memory traffic, NumPy's threads, or a GPU. Each agent's row counts its simulations' kernel calls (`kernel_calls`). For example:

```bash
python examples/agentic-screening/miniapp_screening.py --agents 30 --sim-median 1 \
    --sim-kernel matMulSimple2D --sim-kernel-size 512
```

**A simulation's duration is its cost, not a deadline.** A simulation runs at least as long as the benchmark draws, and a kernel call that ends after it overruns it. Each simulation's result gives the seconds asked (`seconds`) and taken (`seconds_taken`), and each agent's row lists them (`sim_seconds`). The run counts the overruns and gives the largest, but does not fail on them: its results would then depend on the machine's speed.

**Not in this example yet: cache-first admission.** `--admission` accepts only `fcfs`, and says so otherwise.

## Across nodes

A campaign across nodes runs under one Dragon runtime that spans the job's nodes. Give the job its
nodes, one task each (`sbatch -N 2 --ntasks-per-node=1 ...`), and Dragon the site's transport and the
network the nodes reach each other on:

```shell
dragon -w slurm -t tcp --network-prefix '<regex of the nodes' interfaces>' \
    examples/agentic-screening/miniapp_screening.py --runtime dragon \
    --agents 10000 --populations 4 --workers 44 --model openai \
    --endpoints-file FLEET/endpoints --api-key-file FLEET/api-key
```

- The populations are placed on the nodes in contiguous blocks, the mini-app's node first. Each runs
  its simulations on `--workers` workers of its own node, so give each population its share of the
  node's CPUs. This needs RHAPSODY's native backend with `hosts=`.
- RaDEX's dictionary has `--managers` managers on every node.
- The MCP servers and the tool services stay on the mini-app's node.
- The summary records the nodes: `nodes`, and each population's `node`.

`jobs/scale` submits such campaigns at fixed sizes, from 1,000 agents on one CPU node to 300,000 on
30, with the model servers in a GPU job of their own and a report at the end: see
[jobs/README.md](jobs/README.md).

## Recipes

Each recipe loads one resource more than the others. The command lines run as they are; add `--agents`, `--populations` and `--workers` for the size you want.

- **CPU-heavy, the default setting.** Simulations of 10 s median, each spinning a core, two or three per agent; one lookup, and `lookups + 2 × simulations + 2` model calls per agent. The run is simulation-bound: cores set the time.

  ```shell
  python examples/agentic-screening/miniapp_screening.py --agents 1000 --populations 2 --workers 52
  ```

- **GPU-heavy.** Planning calls and long tool results, with short simulations: more model calls per agent, and long contexts that load the GPUs' prefill and KV cache. With vLLM servers:

  ```shell
  python examples/agentic-screening/miniapp_screening.py --model openai \
      --endpoints-file FLEET/endpoints --api-key-file FLEET/api-key \
      --plan-calls 2 --tool-result-tokens 2000 --sim-median 1
  ```

  The recipe can add a thinking budget (`--think-budget 256`), and cache-first admission (`--admission cache`), which is not in this example yet.

- **Model-traffic-heavy.** Planning calls without long results, more lookups and short simulations: many short requests, so the request cap (`--cap`) binds before the token cap, and the agents' own CPU per call starts to count.

  ```shell
  python examples/agentic-screening/miniapp_screening.py --model openai \
      --endpoints-file FLEET/endpoints --api-key-file FLEET/api-key \
      --plan-calls 4 --lookup-calls 3 --sim-median 1
  ```

- **The reasoning trade-off.** A thinking budget B on every decision call, a gain that cuts simulations for the tokens spent, and a cost spread across the candidates, with the budget proportional to each candidate's cost; sweep B against `--sim-median`. More thinking costs model work and cuts simulations: the best B depends on how long the simulations are.

  ```shell
  python examples/agentic-screening/miniapp_screening.py --think-budget 256 --think-gain 1 \
      --sim-cost-sigma 0.5 --think-policy proportional
  ```

- **No GPUs.** The stub model keeps the workflow's shape on CPUs only: its latency per call, replicas per population and a prefix cache.

  ```shell
  python examples/agentic-screening/miniapp_screening.py --model stub --stub-latency 0.05 \
      --stub-replicas 2 --stub-kv-tokens 200000
  ```

  With a thinking budget, `--stub-decode-rate`, `--stub-think-median` and `--stub-think-sigma` shape the stub's thinking.

With the stub, every recipe's per-agent results are those of the benchmark's original LangGraph agents with the same knobs.

## What it writes

The run directory holds:

| Path | Contents |
|---|---|
| `launcher/` | The mini-app's checkpoint and `MiniApp`'s report |
| `populations/` | Each population's checkpoint and stats |
| `app/` | The per-agent rows, the checks and the ranking |
| `otel/` | The agents' OpenTelemetry export |
| `services/` | The MCP servers' and tool services' records |
| `vllm/` | With `--model openai`: the servers' counters over the run |
| `summary.json` | The run's figures |
| `autonmi.json` | The AUTONMI commit the run imported, where it was read, and whether it is the pinned one |

The run's checks:
- every agent verifies;
- the model calls are as the workflow scripts them;
- the simulations' results go to the store and back;
- each simulation ran its seconds: none ended more than 10 ms early; its detail counts the simulations that overran by more than that, and gives the largest overrun;
- each population's tasks equal its agents' simulations;
- the fleet's calls equal the rows';
- with `--model openai`, the tokens equal the servers' own counters.

The script exits 0 when every check passes and the telemetry is complete. `python -m wfMiniAPI.verify RUN/launcher` verifies the run again from its files.

With the stub model, its per-agent results are identical to those of the benchmark's original LangGraph agents, agent for agent.
