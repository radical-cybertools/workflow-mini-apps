#!/usr/bin/env bash
# The ladder's first job: 100 agents of the mini-app under Dragon on one node, with the stub model and
# short simulations, so that a broken environment fails in minutes, before the campaign waits for its
# nodes. jobs/scale submits it (jobs/README.md):
#
#   test.sh RUN
set -u
run=${1:?run directory}
# shellcheck source=/dev/null
source "$AUTONMI_REPO/examples/agentic-screening/jobs/env.sh"
use_framework
echo "test job $SLURM_JOB_ID on $(hostname -s), $(date -Is), commit $AUTONMI_COMMIT, $(nproc) CPUs"
dragon_guard || exit 3
since=$(mktemp -p "$run")
touch "$since"
AUTONMI_MARGIN=60 run_dragon dragon -s "$EXAMPLE/miniapp_screening.py" --runtime dragon --run-dir "$run" \
	--agents 100 --populations 2 --workers 4 --sim-median 0.05 --no-burn --stagger-interval 0.01 --timeout 600
if wait_steps 60; then
	clean_nodes "$since"
fi
rm -f "$since"
echo "the mini-app under dragon exited $status at $(date -Is)"
exit "$status"
