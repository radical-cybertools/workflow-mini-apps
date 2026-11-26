#!/usr/bin/env bash
# The ladder's model job: one vLLM server per GPU of the job (or per AUTONMI_TP GPUs, tensor
# parallel, for a model larger than one GPU), on one GPU node. jobs/scale submits it (jobs/README.md):
#
#   model.sh FLEET [GPUS]
#
# GPUS: how many of the job's GPUs to use, all by default; the job fails when it has fewer. Writes into
# FLEET: `endpoints` (one base URL per server, once all answer), `api-key` (readable by the owner
# only), `gpu.csv` (nvidia-smi every second, where there is one, for the record) and `vllm-<i>.log`.
# Ends when FLEET/stop appears. The servers' host in `endpoints` is AUTONMI_ENDPOINT_HOST, where %h
# is the node's short name and %f its full one (default %h): `%h-ib` names another network.
#
# A server that dies is started again on its own port, up to AUTONMI_VLLM_RESTARTS times (3); past
# that the job ends. The job is requeueable (jobs/scale submits it with --requeue): requeued, it starts
# over with the same key on new ports, appending to its logs, and the mini-app waits for it.
#
# The model's settings, from the site's settings or the environment, Qwen3-8B's by default:
# AUTONMI_MODEL (the weights), AUTONMI_MODEL_NAME (the served name, Qwen3-8B), AUTONMI_TOOL_PARSER
# (hermes), AUTONMI_REASONING (a reasoning parser; unset: thinking off), AUTONMI_VLLM_ARGS (more of
# vLLM's options), AUTONMI_GPU_VARIABLE (the variable that names the GPUs, CUDA_VISIBLE_DEVICES).
set -u
fleet=${1:?fleet directory}
want=${2:-}
# shellcheck source=/dev/null
source "$AUTONMI_REPO/examples/agentic-screening/jobs/env.sh"
use_model
echo "model job $SLURM_JOB_ID on $(hostname -s), $(date -Is), commit $MINIAPP_COMMIT, start ${SLURM_RESTART_COUNT:-0}"
var=${AUTONMI_GPU_VARIABLE:-CUDA_VISIBLE_DEVICES}
listed=${!var:-}
if [ -z "$listed" ] && command -v nvidia-smi > /dev/null; then
	listed=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
fi
IFS=, read -ra gpus <<< "${listed:-0}"
if [ -n "$want" ]; then
	if [ "$want" -gt "${#gpus[@]}" ]; then
		echo "asked for $want GPUs, but the job has ${#gpus[@]} ($var=${listed:-none}): stopping"
		exit 1
	fi
	gpus=("${gpus[@]:0:$want}")
fi
command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
tp=${AUTONMI_TP:-1}
if [ "$tp" -lt 1 ] || [ $((${#gpus[@]} % tp)) -ne 0 ]; then
	echo "AUTONMI_TP=$tp does not divide the job's ${#gpus[@]} GPUs"
	exit 1
fi
devices=()
for ((g = 0; g < ${#gpus[@]} / tp; g++)); do
	devices[g]=$(
		IFS=,
		echo "${gpus[*]:$((g * tp)):$tp}"
	)
done
parallel=()
[ "$tp" -gt 1 ] && parallel=(--tensor-parallel-size "$tp")
echo "servers: ${#devices[@]}, GPUs per server: $tp (${devices[*]}), pinned with $var"
[ -s "$fleet/api-key" ] || (umask 077 && python -c 'import secrets; print(secrets.token_hex(24))' > "$fleet/api-key")
VLLM_API_KEY=$(cat "$fleet/api-key")
export VLLM_API_KEY
rm -f "$fleet/stop" "$fleet/endpoints" "$fleet/endpoints.tmp"
smi=
if command -v nvidia-smi > /dev/null; then
	nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 >> "$fleet/gpu.csv" &
	smi=$!
fi
reasoning=()
if [ -n "${AUTONMI_REASONING:-}" ]; then
	reasoning=(--reasoning-parser "$AUTONMI_REASONING" --reasoning-config
		"{\"reasoning_start_str\": \"${AUTONMI_REASONING_START:-<think>}\", \"reasoning_end_str\": \"${AUTONMI_REASONING_END:-</think>}\"}")
	echo "reasoning: ${reasoning[*]}"
fi
read -ra extra <<< "${AUTONMI_VLLM_ARGS:-}"
host=${AUTONMI_ENDPOINT_HOST:-%h}
host=${host//%h/$(hostname -s)}
host=${host//%f/$(hostname -f)}
pids=()
ports=()
deaths=()
stop_all() {
	for p in "${pids[@]}"; do kill "$p" 2> /dev/null; done
	for p in "${pids[@]}"; do wait "$p" 2> /dev/null; done
	[ -z "$smi" ] || kill "$smi" 2> /dev/null
}
start_server() { # index
	local i=$1
	echo "=== start $(date -Is), job $SLURM_JOB_ID, start ${SLURM_RESTART_COUNT:-0}" >> "$fleet/vllm-$i.log"
	env "$var=${devices[$i]}" vllm serve "${AUTONMI_MODEL:?set in the site settings}" \
		--served-model-name "${AUTONMI_MODEL_NAME:-Qwen3-8B}" --host 0.0.0.0 --port "${ports[$i]}" \
		--enable-prefix-caching --enable-auto-tool-choice --tool-call-parser "${AUTONMI_TOOL_PARSER:-hermes}" \
		--enable-prompt-tokens-details --gpu-memory-utilization 0.9 --max-model-len 32768 \
		"${parallel[@]}" "${reasoning[@]}" "${extra[@]}" >> "$fleet/vllm-$i.log" 2>&1 &
	pids[i]=$!
}
wait_ready() { # index; fails if the server exits first
	local i=$1
	until curl -sf "http://127.0.0.1:${ports[$i]}/health" > /dev/null; do
		kill -0 "${pids[$i]}" 2> /dev/null || return 1
		sleep 2
	done
}
free_port() { python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])'; }
# One server at a time, so that they do not race on vLLM's compile cache. A port is free when it is
# picked, but another process can take it before vLLM binds it: then the server starts again on
# another, at most twice.
for i in "${!devices[@]}"; do
	deaths[i]=0
	t0=$(date +%s)
	for try in 1 2 3; do
		ports[i]=$(free_port)
		start_server "$i"
		wait_ready "$i" && break
		if [ "$try" -lt 3 ] && grep -qi "address already in use" "$fleet/vllm-$i.log"; then
			echo "vLLM $i: port ${ports[$i]} was taken; trying another"
			continue
		fi
		echo "vLLM $i exited before it was ready"
		tail -30 "$fleet/vllm-$i.log"
		stop_all
		exit 1
	done
	echo "http://$host:${ports[$i]}" >> "$fleet/endpoints.tmp"
	echo "vLLM $i (devices ${devices[$i]}) ready in $(($(date +%s) - t0)) s on port ${ports[$i]}"
	grep "GPU KV cache size" "$fleet/vllm-$i.log" | tail -1 || true
done
mv "$fleet/endpoints.tmp" "$fleet/endpoints"
while [ ! -e "$fleet/stop" ]; do
	for i in "${!pids[@]}"; do
		kill -0 "${pids[$i]}" 2> /dev/null && continue
		wait "${pids[$i]}"
		code=$?
		deaths[i]=$((deaths[i] + 1))
		echo "vLLM $i exited with $code at $(date -Is) (${deaths[$i]} of ${AUTONMI_VLLM_RESTARTS:-3} restarts)"
		tail -20 "$fleet/vllm-$i.log"
		if [ "${deaths[$i]}" -gt "${AUTONMI_VLLM_RESTARTS:-3}" ]; then
			echo "vLLM $i died too often; stopping"
			stop_all
			exit 1
		fi
		t0=$(date +%s)
		start_server "$i"
		wait_ready "$i" && echo "vLLM $i back in $(($(date +%s) - t0)) s on port ${ports[$i]}"
	done
	sleep 5
done
echo "stopping at $(date -Is)"
stop_all
