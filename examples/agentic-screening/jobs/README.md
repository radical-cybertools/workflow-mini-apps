# The scale ladder

`jobs/scale` runs the screening campaign at a fixed size, from 1,000 agents on one CPU node to
300,000 on 30, as one chain of Slurm jobs. The settings of each site, its paths, partitions and
nodes, are in a file of their own, so the same ladder runs on any Slurm cluster.

## Quick start

From `examples/agentic-screening/`:

```bash
jobs/scale dry-run 10k stub   # print the jobs, submit nothing
jobs/scale submit  10k stub   # submit them and return at once: safe to log out
jobs/scale status  10k stub   # the jobs' states, and the run's report once it is done
jobs/scale cancel  10k stub   # cancel them
```

With `stub`, the stub model answers and no GPU job is submitted: it checks the CPU side first.
Without it, vLLM servers on a GPU node answer. Start small: `1k stub`, `10k stub`, then `10k`,
`100k` and `300k`.

## The rungs

| Rung | Agents | CPU nodes | Populations × workers | GPUs, one vLLM server each | Requests in flight per server | MCP servers | Campaign's time limit | Simulations' bound |
|---|---|---|---|---|---|---|---|---|
| `1k` | 1,000 | 1 | 2 × 44 | 1 | 96 | 4 | 0:40 | 6 min |
| `10k` | 10,000 | 2 | 4 × 44 | 2 | 96 | 4 | 1:15 | 30 min |
| `100k` | 100,000 | 10 | 20 × 44 | 2 | 320 | 4 | 2:00 | 60 min |
| `300k` | 300,000 | 30 | 60 × 44 | 8 | 960 | 8 | 2:00 | 60 min |

- **The table is for CPU nodes of 96 CPUs.** Each node has 2 populations of up to 5,000 agents, and
  each population 44 simulation workers of one CPU: 88 workers, and 8 CPUs for the rest. Each node
  also has 4 of the store's managers. The first node also runs the MCP servers and the tool
  services.
- **On other nodes the workers follow the site's CPUs.** With `AUTONMI_NODE_CPUS=64`, a population has
  (64 − 8) / 2 = 28 workers. The rung keeps its agents and nodes. With fewer workers than the table's,
  the simulations take longer, so the campaign's time limits grow by as much: `10k` on two nodes of 64
  CPUs has 112 workers, a bound of 47 minutes and a limit of 1:58. `jobs/scale dry-run` prints the
  run's shape, its bound and its limits.
- **The GPUs** are all on one node, in their own job. Each agent always uses the same server.
- **The bound** is the simulation work over the workers: 31.45 core-seconds per agent with the
  benchmark's default settings, over every population's workers. No run can finish sooner.
- **Requests in flight** are per server, shared by the populations: 16 per population and server at
  `100k` and `300k`.

## Set up once

1. **The environment.** A Python 3.12 environment with the AUTONMI framework at the commit the example
   pins, as the example's README says, with Dragon and RaDEX. The jobs start Dragon as `python -m dragon` with its python, and put
   this tree's `src` first on `PYTHONPATH`, so the run uses this tree's wfMiniAPI. Several nodes need RHAPSODY's native backend with `hosts=`, which
   keeps each population's workers on its node. The model job needs vLLM, in the same environment or
   another one (`AUTONMI_MODEL_VENV`).
2. **The site's settings.** Copy `jobs/local.env.example` to `jobs/local.env` (git ignores it) and fill
   it in. To keep several sites' files, put each anywhere and name it in `AUTONMI_LOCAL_ENV`:
   `AUTONMI_LOCAL_ENV=~/sites/mycluster.env jobs/scale submit 10k`. The jobs source the file in bash,
   so it can also set `PATH` or load modules.

| Setting | What | Default |
|---|---|---|
| `AUTONMI_VENV` | the framework's environment | required |
| `AUTONMI_RUNS` | where the runs go, on a file system the nodes share; an absolute path | required |
| `AUTONMI_OVERLAY` | directories put on `PYTHONPATH` after this tree's `src` | none |
| `SLURM_ACCOUNT` | the jobs' account | none |
| `SLURM_CPU_PARTITION`, `SLURM_GPU_PARTITION` | the partitions of the CPU jobs and of the model job | required |
| `SLURM_GPU_GRES` | the model job's GPUs, ending in a count (`gpu:1`, `gpu:<type>:1`), which the rung's GPUs replace; empty for no `--gres` | none |
| `AUTONMI_QOS_TEST`, `AUTONMI_QOS_CAMPAIGN`, `AUTONMI_QOS_MODEL` | the QOS of the small jobs, the campaign and the model job | none |
| `AUTONMI_SBATCH_CAMPAIGN`, `AUTONMI_SBATCH_MODEL` | more sbatch options for the campaign and the model job, such as `--exclude` or `--reservation` | none |
| `AUTONMI_NODE_MEM` | memory to ask for on each CPU node, such as `300G` | required |
| `AUTONMI_NODE_CPUS` | CPUs to ask for on each CPU node | 96 |
| `AUTONMI_NODE_EXCLUSIVE` | `0` to share the nodes, where a limit on CPUs per user forbids whole ones | 1 |
| `AUTONMI_NODE_POPULATIONS`, `AUTONMI_NODE_RESERVE`, `AUTONMI_NODE_WORKERS` | populations per node, CPUs left for the rest, and workers per population | 2, 8, and the CPUs left over the populations |
| `AUTONMI_DRAGON_ARGS` | Dragon's launcher options across nodes, a bash array: the workload manager, the transport, the network. Needed from `10k` up | none |
| `AUTONMI_MODEL_VENV`, `AUTONMI_MODEL`, `AUTONMI_MODEL_NAME` | the model job's environment, the model's weights, and its served name | the framework's, none, Qwen3-8B |
| `AUTONMI_GPU_CPUS`, `AUTONMI_GPU_MEM_GB` | CPUs and memory per GPU for the model job | 12 and 110 |
| `AUTONMI_CAP_TOKENS` | tokens of live context in flight per server, for the GPU's memory | the mini-app's |
| `AUTONMI_CAMPAIGN_ARGS` | more of the mini-app's options, for every campaign | none |

`jobs/local.env.example` lists the model servers' own settings too (`AUTONMI_TOOL_PARSER`,
`AUTONMI_REASONING`, `AUTONMI_TP`, ...).

**For one run,** set these in front of the command:

| Variable | What |
|---|---|
| `NODES`, `GPUS` | the CPU nodes and the GPUs, in place of the rung's |
| `TIME`, `TIMEOUT`, `MODEL_TIME` | the campaign's time limit, the seconds its populations may run, and the model job's time limit |
| `ARGS` | more of the mini-app's options |
| `TAG` | a name for the run, beside the rung's, so that it does not take the rung's directories |

For example, the `10k` rung on one node, beside the usual two:
`NODES=1 TAG=1n jobs/scale submit 10k stub`, then `TAG=1n jobs/scale status 10k stub`.

## What it submits

1. **A test.** 100 agents of the mini-app under Dragon on one node, with short simulations. If it
   fails, the rest is cancelled.
2. **The model job** on one GPU node, one vLLM server per GPU.
3. **The campaign,** under one Dragon runtime across the CPU nodes. It starts once the test has
   passed and the model job runs, and waits up to 30 minutes for its servers.
4. **The report,** after the campaign however it ends. It also ends the model job, so that it doesn't
   hold its GPUs.

With `stub`, only the test, the campaign and the report. The model job waits with its GPUs while the
campaign waits for its CPU nodes, so its time limit covers both. If that wait is longer, raise
`MODEL_TIME`.

**Dragon jobs, one per node.** The test and the campaign are named `autonmi-dragon`, and wait for
each other's end (Slurm's `singleton`), across rungs too; `AUTONMI_DRAGON_SINGLETON=0` turns that
off. Each also stops at once if another running job of yours with "dragon" in its name is on one of
its nodes. Each ends its
run a few minutes before its time limit, then cleans up after Dragon on its nodes: your files in
`/dev/shm`, and an MPS daemon that Dragon's clean-up can start on nodes with NVIDIA GPUs.

## Results

Everything is under `AUTONMI_RUNS`:

| Path | What |
|---|---|
| `scale-10k/scale-report.json` | the report: every agent verified, and where the time went |
| `scale-10k/summary.json` | the mini-app's results and checks; `ok` is true when every check passes |
| `scale-10k/app/` | the agents' rows, the checks and the ranking |
| `scale-10k/campaign-*.out`, `report-*.out`, and Dragon's logs | the jobs' output |
| `scale-10k/timeline`, `jobs.txt` | when the campaign job's phases began, and the chain's accounting |
| `scale-10k-fleet/` | the model job's output and the servers' logs |
| `scale-10k-test/` | the test's run and output |
| `scale-10k.jobs` | the job ids, the commit and the settings' file |

To run a rung again, move `scale-10k*` away first.

**The report** (`jobs/scale status`, or `scale-report.json`) says:
- whether every agent verifies. It verifies each agent's report as its row recorded it, with the
  benchmark's own verification, so it works when the mini-app wrote no summary. It gives the run's
  own checks beside it;
- where the time went: how long each job waited in the queue and ran; the campaign job's start-up,
  agents, wind-down and clean-up; the agents' span against the simulations' bound; and the agents'
  own time, in simulations, in model calls, and waiting for a worker;
- the nodes, with their populations and agents;
- the AUTONMI commit the run imported, and whether it is the one the example pins.

**Comparing two runs.** With the same seed, agent *i* is the same candidate at every size. With the
stub model, its simulations, model calls and report are the same on one node as on several:

```bash
python jobs/scale_report.py compare RUNS/scale-1k-stub RUNS/scale-10k-stub
```

checks the agents both runs have, agent by agent, and exits 0 when they all match.

## If something goes wrong

- **The test failed.** Its output is `scale-<rung>-test/test-*.out`, and its run in that directory.
  The rest was cancelled.
- **The campaign wrote no `summary.json`.** The report still verifies the agents from the rows of the
  populations that ended, and names the ones that did not. `jobs/scale report <rung>` writes it again
  where you are. It reads every agent's row, so run it on a compute node if your login nodes are
  not for that.
- **A population never ends.** The mini-app waits `TIMEOUT` seconds for its populations, then stops
  them. The campaign job also ends its run 4 minutes before its time limit.
- **`... is not set`.** Add the setting to the site's file.
