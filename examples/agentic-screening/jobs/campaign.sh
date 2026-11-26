#!/usr/bin/env bash
# shellcheck disable=SC2154 # dragon_args and status: set by env.sh's functions
# The ladder's campaign job: the screening mini-app under one Dragon runtime, on the job's node or
# nodes. jobs/scale submits it (jobs/README.md):
#
#   campaign.sh RUN FLEET MODEL_JOB [miniapp_screening.py options]
#
# FLEET is the model job's directory, whose `stop` file this job writes when the mini-app ends, and
# MODEL_JOB that job's id; both `none` with the stub. With a model job, the campaign starts only once
# that job runs and all its servers answer, within AUTONMI_MODEL_WAIT seconds, and fails at once when
# the model job is not running. One node runs `dragon -s`; several run `dragon` with
# AUTONMI_DRAGON_ARGS, the site's transport and network. Dragon runs as `python -m dragon` with the
# framework's python, which then starts the mini-app with that same python. The job writes
# RUN/timeline (when it started, when Dragon started and ended, when it ended) for the report, and
# exits 0 when the run's summary says ok.
set -u
run=${1:?run directory}
fleet=${2:?fleet directory, or none}
model_job=${3:?model job id, or none}
shift 3
# shellcheck source=/dev/null
source "$AUTONMI_REPO/examples/agentic-screening/jobs/env.sh"
use_framework
mark() { echo "$1 $(date +%s.%N)" >> "$run/timeline"; }
mark job_start
nodes=${SLURM_JOB_NUM_NODES:-1}
echo "campaign job $SLURM_JOB_ID on $nodes node(s): $(job_nodes | paste -sd' ' -), $(date -Is), commit $MINIAPP_COMMIT, $(nproc) CPUs here"
dragon_guard || exit 3
if [ "$model_job" != none ] && ! wait_for_model "$model_job" "$fleet" "${AUTONMI_MODEL_WAIT:-1800}"; then
	mark job_end
	exit 4
fi
dragon_args_here
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
summary_ok "$run"
