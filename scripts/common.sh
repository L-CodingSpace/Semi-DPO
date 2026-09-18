# Shared settings for the launch scripts.  Source this file; do not run it.
#
# Multi-node runs: set NUM_MACHINES, NUM_PROCESSES (total GPUs over all
# machines), MACHINE_RANK and MAIN_PROCESS_IP on every machine.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

NUM_MACHINES=${NUM_MACHINES:-1}
NUM_PROCESSES=${NUM_PROCESSES:-8}
MACHINE_RANK=${MACHINE_RANK:-0}
MAIN_PROCESS_IP=${MAIN_PROCESS_IP:-127.0.0.1}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}

MODEL=${MODEL:-sd15}  # sd15 | sdxl
case "${MODEL}" in
  sd15) ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-configs/multi_gpu.yaml} ;;
  sdxl) ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-configs/fsdp.yaml} ;;
  *) echo "MODEL must be sd15 or sdxl, got '${MODEL}'" >&2; exit 1 ;;
esac

# launch <accelerate config> <script> [args...]
launch() {
  local config=$1
  shift
  accelerate launch --config_file "${config}" \
    --num_machines "${NUM_MACHINES}" --num_processes "${NUM_PROCESSES}" \
    --machine_rank "${MACHINE_RANK}" \
    --main_process_ip "${MAIN_PROCESS_IP}" --main_process_port "${MAIN_PROCESS_PORT}" \
    "$@"
}
