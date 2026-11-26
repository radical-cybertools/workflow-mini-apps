#!/usr/bin/env bash
# The ladder's last job, after the campaign however it ended: the run's report, and the model job's
# end. jobs/scale submits it (jobs/README.md):
#
#   report.sh RUN FLEET JOBS      JOBS: test=<id>,model=<id or empty>,campaign=<id>
#
# It writes the stop file the model job waits for (the campaign writes it too, if it got that far),
# keeps the chain's accounting in RUN/jobs.txt, writes RUN/scale-report.json, and cancels the model
# job if it has not ended two minutes later.
set -u
run=${1:?run directory}
fleet=${2:?fleet directory, or none}
jobs=${3:?the job ids of the chain}
# shellcheck source=/dev/null
source "$AUTONMI_REPO/examples/agentic-screening/jobs/env.sh"
use_framework
echo "report job $SLURM_JOB_ID on $(hostname -s), $(date -Is), commit $AUTONMI_COMMIT: $jobs"
[ "$fleet" = none ] || touch "$fleet/stop"
model=$(tr ',' '\n' <<< "$jobs" | sed -n 's/^model=//p')
ids=$(tr ',' '\n' <<< "$jobs" | cut -d= -f2 | grep . | paste -sd, -)
sacct -X -P -j "$ids" -o JobID,JobName,State,Submit,Start,End,Elapsed,NNodes,NodeList > "$run/jobs.txt" 2> /dev/null ||
	echo "no accounting: sacct failed"
python "$JOBS/scale_report.py" write "$run" --jobs "$jobs"
status=$?
if [ -n "$model" ]; then
	for _ in $(seq 24); do
		squeue -h -j "$model" 2> /dev/null | grep -q . || break
		sleep 5
	done
	if squeue -h -j "$model" 2> /dev/null | grep -q .; then
		scancel "$model" && echo "cancelled the model job $model: it had not ended on its stop file"
	fi
fi
exit $status
