# shellcheck shell=bash disable=SC2034 # the scripts that source it use what it sets
# Sourced by the ladder's job scripts: the site's settings, the environments, the Dragon guard, the
# wait for the model servers and the clean-up after Dragon. jobs/scale exports AUTONMI_REPO and
# AUTONMI_LOCAL_ENV, and submits with --export=ALL. Needs bash 4.4 or newer.
if ((BASH_VERSINFO[0] * 100 + BASH_VERSINFO[1] < 404)); then
	echo "the ladder's jobs need bash 4.4 or newer (this is $BASH_VERSION)" >&2
	exit 1
fi
# shellcheck source=/dev/null
source "${AUTONMI_LOCAL_ENV:?set by jobs/scale}"
EXAMPLE=${AUTONMI_REPO:?set by jobs/scale}/examples/agentic-screening
JOBS=$EXAMPLE/jobs
# This tree's commit (workflow-mini-apps'), for the logs: a tree unpacked from `git archive` names it
# in .git-commit. The AUTONMI commit a run imports is in its autonmi.json.
MINIAPP_COMMIT=$(cat "$AUTONMI_REPO/.git-commit" 2> /dev/null || git -C "$AUTONMI_REPO" rev-parse --short HEAD 2> /dev/null || echo unknown)
export MINIAPP_COMMIT
if [ -n "${AUTONMI_CUDART:-}" ]; then
	export LD_LIBRARY_PATH="$AUTONMI_CUDART${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
ulimit -n "$(ulimit -Hn)"

# The framework's environment, with this tree's wfMiniAPI first, then the site's overlays
use_framework() {
	# shellcheck source=/dev/null
	source "${AUTONMI_VENV:?set in the site settings}/bin/activate"
	export PYTHONPATH="$AUTONMI_REPO/src${AUTONMI_OVERLAY:+:$AUTONMI_OVERLAY}"
}

# The model servers' environment (vLLM), the framework's unless the site names another
use_model() {
	# shellcheck source=/dev/null
	source "${AUTONMI_MODEL_VENV:-${AUTONMI_VENV:?set in the site settings}}/bin/activate"
	unset PYTHONPATH
}

# Dragon's launcher options for this job: `-s` on one node, the site's AUTONMI_DRAGON_ARGS on several
dragon_args_here() {
	if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then
		dragon_args=(${AUTONMI_DRAGON_ARGS[@]+"${AUTONMI_DRAGON_ARGS[@]}"})
	else
		dragon_args=(-s)
	fi
}

# The summary's verdict on the run in $1: 0 when it says ok. `dragon` can exit 0 when its head
# process failed, so the summary decides.
summary_ok() {
	python -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))['ok'] else 1)" "$1/summary.json" 2> /dev/null
}

# Until every server in $2/endpoints answers /health, while the model job $1 runs; at most $3
# seconds. Fails at once when the model job is not running (it ended, failed, was cancelled or
# requeued), so that the campaign's nodes do not wait on servers that will not come.
wait_for_model() { # model-job fleet seconds
	local job=$1 fleet=$2 end=$(($(date +%s) + $3)) state
	while :; do
		state=$(squeue -h -j "$job" -o %T 2> /dev/null || true)
		case $state in
		RUNNING | CONFIGURING) ;;
		*)
			echo "the model job $job is ${state:-no longer in the queue}: not starting the campaign"
			return 1
			;;
		esac
		if python - "$fleet/endpoints" << 'EOF'; then
import sys
import urllib.request

try:
    urls = [u.strip() for u in open(sys.argv[1]) if u.strip()]
except OSError:
    sys.exit(1)
for url in urls:
    try:
        urllib.request.urlopen(url + "/health", timeout=5).close()
    except Exception:
        sys.exit(1)
sys.exit(0 if urls else 1)
EOF
			echo "the model job's servers answer: $(paste -sd' ' "$fleet/endpoints")"
			return 0
		fi
		if [ "$(date +%s)" -ge "$end" ]; then
			echo "the model job's servers did not all answer within $3 s: not starting the campaign"
			return 1
		fi
		sleep "${AUTONMI_MODEL_POLL:-10}"
	done
}

job_nodes() { scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname -s)}" 2> /dev/null || hostname -s; }

# One Dragon runtime per node: fail if another running job of ours on one of this job's nodes has
# "dragon" in its name.
dragon_guard() {
	local n others
	for n in $(job_nodes); do
		others=$(squeue -u "$USER" -w "$n" -t RUNNING -h -o '%i %j' | awk -v me="$SLURM_JOB_ID" '$1 != me && $2 ~ /dragon/')
		if [ -n "$others" ]; then
			echo "another Dragon job of ours runs on $n: $others; stopping" >&2
			return 1
		fi
	done
}

# Until no step of this job but the batch and extern steps is left (Dragon's, under -w slurm), at most
# $1 seconds.
wait_steps() {
	local end=$(($(date +%s) + $1))
	while squeue -h -s -j "$SLURM_JOB_ID" -o '%i %j' 2> /dev/null | grep -v -E '\.(batch|extern) ' | grep -q .; do
		if [ "$(date +%s)" -ge "$end" ]; then
			echo "steps still running after $1 s: $(squeue -h -s -j "$SLURM_JOB_ID" -o '%i %j' | tr '\n' ' ')"
			return 1
		fi
		sleep 5
	done
}

# After Dragon has exited, on every node of the job, each step bounded by `timeout`: quit an MPS
# daemon of ours (Dragon's clean-up after an abnormal exit starts one on nodes with NVIDIA GPUs, and
# it outlives the job), remove our files from /tmp/nvidia-mps, and remove the /dev/shm files of ours
# newer than $1 when no other job of ours runs on the node. Only call it once wait_steps has passed:
# never `srun --overlap` into a running Dragon job.
clean_nodes() {
	local nodes=${SLURM_JOB_NUM_NODES:-1}
	timeout -k 10 120 srun -N "$nodes" --ntasks-per-node=1 --overlap bash -c '
		h=$(hostname -s)
		if pgrep -u "$USER" -x nvidia-cuda-mps >/dev/null 2>&1; then
			echo quit | timeout 10 nvidia-cuda-mps-control; echo "$h: quit an MPS daemon of ours"
		fi
		find /tmp/nvidia-mps -maxdepth 1 -user "$USER" ! -path /tmp/nvidia-mps -delete 2>/dev/null
		others=$(squeue -u "$USER" -w "$h" -h -o %i | grep -vc "^$SLURM_JOB_ID$" || true)
		if [ "$others" -eq 0 ]; then
			timeout 60 find /dev/shm -maxdepth 1 -user "$USER" -newer "$1" -delete 2>/dev/null
		else
			echo "$h: /dev/shm left as it is: $others other job(s) of ours here"
		fi' _ "$1" < /dev/null || echo "the clean-up of the nodes did not finish (status $?)"
}

# A Dragon run that ends before the job's limit: AUTONMI_MARGIN seconds (240) before it, or at once on
# Slurm's USR1 (--signal=B:USR1@240) or a cancel's TERM. `timeout` signals the run's whole process
# group: TERM, then KILL a minute later. Sets `status`.
run_dragon() { # command...
	local end_s left child
	end_s=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s 2> /dev/null || echo $(($(date +%s) + 86400)))
	left=$((end_s - ${AUTONMI_MARGIN:-240} - $(date +%s)))
	if [ "$left" -lt 60 ]; then
		echo "only $left s left before the job's limit: not starting"
		status=124
		return
	fi
	echo "$(date -Is): starting, at most $left s: $*"
	timeout -k 60 "$left" "$@" < /dev/null &
	child=$!
	trap 'echo "USR1 at $(date -Is): ending the run"; kill -TERM '"$child"' 2>/dev/null' USR1
	trap 'echo "TERM at $(date -Is): ending the run"; kill -TERM '"$child"' 2>/dev/null' TERM
	wait "$child"
	status=$?
	while kill -0 "$child" 2> /dev/null; do # a trapped signal ends a wait early
		wait "$child"
		status=$?
	done
	trap - USR1 TERM
	echo "$(date -Is): exited $status"
}
