#!/usr/bin/env bash
# The ladder's campaign job: the screening mini-app under one Dragon runtime, on the job's node or
# nodes. jobs/scale submits it (jobs/README.md):
#
#   campaign.sh RUN FLEET [miniapp_screening.py options]
#
# FLEET is the model job's directory, whose `stop` file this job writes when the mini-app ends; `none`
# with the stub. One node runs `dragon -s`; several run `dragon` with AUTONMI_DRAGON_ARGS, the site's
# transport and network. Dragon runs as `python -m dragon` with the framework's python, which then
# starts the mini-app with that same python. The job writes RUN/timeline (when it started, when Dragon started and
# ended, when it ended) for the report, and exits 0 when the run's summary says ok.
set -u
run=${1:?run directory}
fleet=${2:?fleet directory, or none}
shift 2
# shellcheck source=/dev/null
source "$AUTONMI_REPO/examples/agentic-screening/jobs/env.sh"
use_framework
mark() { echo "$1 $(date +%s.%N)" >> "$run/timeline"; }
mark job_start
nodes=${SLURM_JOB_NUM_NODES:-1}
echo "campaign job $SLURM_JOB_ID on $nodes node(s): $(job_nodes | paste -sd' ' -), $(date -Is), commit $AUTONMI_COMMIT, $(nproc) CPUs here"
dragon_guard || exit 3
if [ "$nodes" -gt 1 ]; then
	dragon_args=(${AUTONMI_DRAGON_ARGS[@]+"${AUTONMI_DRAGON_ARGS[@]}"})
else
	dragon_args=(-s)
fi
echo "dragon ${dragon_args[*]}"
since=$run/.since-$SLURM_JOB_ID # on the shared file system, so that every node's clean-up reads it
touch "$since"
mark dragon_start
run_dragon python -m dragon "${dragon_args[@]}" "$EXAMPLE/miniapp_screening.py" --runtime dragon --run-dir "$run" "$@"
mark dragon_end
[ "$fleet" = none ] || touch "$fleet/stop" # the model job ends; the report job makes sure
if wait_steps 120; then
	clean_nodes "$since"
else
	echo "not cleaning the nodes: a step of this job is still running"
fi
rm -f "$since"
mark job_end
# `dragon` can exit 0 when its head process failed: the summary decides
python -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))['ok'] else 1)" "$run/summary.json" 2> /dev/null
