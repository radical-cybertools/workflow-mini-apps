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

The physics, the prompts, the benchmark and the verification need nothing outside the package.

## What it needs

- **The AUTONMI framework.** It runs populations of agents as nodes of an AsyncFlow graph, on Flowgentic's agent loop. It is installed separately: the RADICAL team gives access to it and its install recipe. It brings Flowgentic, AsyncFlow, RHAPSODY and the OpenTelemetry SDK.
- **This repository's `wfMiniAPI`,** with `MiniApp` for runs of many engines and components.
- **Under Dragon:** Dragon and RaDEX, as RHAPSODY's native backend uses them.

Without the framework, this repository's tests skip this example, apart from the benchmark's own test and the contract check, which reads its source.

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

`--help` lists the knobs, among them:
- `--agents`, `--seed`, `--populations`, `--workers`;
- the simulations' duration (`--sim-median`, `--no-burn` to sleep instead of using a CPU);
- `--lookup-calls`;
- the model servers' admission caps;
- the telemetry's span sampling (`--span-ratio 1` keeps every span);
- `--timeout`.

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

The run's checks:
- every agent verifies;
- the model calls are as the workflow scripts them;
- the simulations' results go to the store and back;
- each population's tasks equal its agents' simulations;
- the fleet's calls equal the rows';
- with `--model openai`, the tokens equal the servers' own counters.

The script exits 0 when every check passes and the telemetry is complete. `python -m wfMiniAPI.verify RUN/launcher` verifies the run again from its files.

With the stub model, its per-agent results are identical to those of the benchmark's original LangGraph agents, agent for agent.
