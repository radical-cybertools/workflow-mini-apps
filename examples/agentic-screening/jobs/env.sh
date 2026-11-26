# Sourced by the ladder's job scripts: the site's settings, the environments, the Dragon guard and the
# clean-up after Dragon. jobs/scale exports AUTONMI_REPO and AUTONMI_LOCAL_ENV, and submits with
# --export=ALL.
# shellcheck source=/dev/null
source "${AUTONMI_LOCAL_ENV:?set by jobs/scale}"
EXAMPLE=${AUTONMI_REPO:?set by jobs/scale}/examples/agentic-screening
JOBS=$EXAMPLE/jobs
AUTONMI_COMMIT=$(git -C "$AUTONMI_REPO" rev-parse --short HEAD 2> /dev/null || cat "$AUTONMI_REPO/.git-commit" 2> /dev/null || echo unknown)
export AUTONMI_COMMIT
export LD_LIBRARY_PATH="${AUTONMI_CUDART:+$AUTONMI_CUDART:}${LD_LIBRARY_PATH:-}"
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
