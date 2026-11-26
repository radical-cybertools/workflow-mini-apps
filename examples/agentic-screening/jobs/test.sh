#!/usr/bin/env bash
# shellcheck disable=SC2154 # dragon_args and status: set by env.sh's functions
# The ladder's first job: 100 agents of the mini-app under Dragon, with the stub model and short
# simulations, so that a broken environment fails in minutes, before the campaign waits for its
# nodes. On one node it runs `dragon -s`; for a rung of several nodes, jobs/scale gives it two, and it
# runs `dragon` with the campaign's AUTONMI_DRAGON_ARGS, so that a wrong transport or network fails
# here too. It passes only when the run's summary says ok. jobs/scale submits it (jobs/README.md):
#
#   test.sh RUN
set -u
run=${1:?run directory}
# shellcheck source=/dev/null
source "$AUTONMI_REPO/examples/agentic-screening/jobs/env.sh"
use_framework
echo "test job $SLURM_JOB_ID on $(job_nodes | paste -sd' ' -), $(date -Is), commit $MINIAPP_COMMIT, $(nproc) CPUs"
dragon_guard || exit 3
dragon_args_here
echo "dragon ${dragon_args[*]}"
since=$(mktemp "$run/.since.XXXXXX")
touch "$since"
AUTONMI_MARGIN=60 run_dragon python -m dragon "${dragon_args[@]}" "$EXAMPLE/miniapp_screening.py" --runtime dragon \
	--run-dir "$run" --agents 100 --populations 2 --workers 4 --sim-median 0.05 --no-burn --stagger-interval 0.01 \
	--timeout 600
if wait_steps 60; then
	clean_nodes "$since"
fi
rm -f "$since"
echo "the mini-app under dragon exited $status at $(date -Is)"
if ! summary_ok "$run"; then
	echo "the test failed: its summary ($run/summary.json) is missing or not ok"
	exit 1
fi
echo "the test passed"
