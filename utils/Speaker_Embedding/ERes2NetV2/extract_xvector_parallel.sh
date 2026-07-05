#!/bin/bash
# Extract x-vector in parallel across multiple GPUs
#
# Usage:
#   ./extract_xvector_parallel.sh <gpu_list> <procs_per_gpu> <dataset> [extra_args...]
#
# Example:
#   ./extract_xvector_parallel.sh "0,1,2,3" 2 AISHELL-2
#   ./extract_xvector_parallel.sh "2,3" 1 MAGICDATA-cut --overwrite
#
# Args:
#   gpu_list: Comma-separated GPU IDs (e.g., "0,1,2,3")
#   procs_per_gpu: Number of processes per GPU
#   dataset: Dataset name (e.g., AISHELL-2, MAGICDATA-cut)
#   extra_args: Additional arguments passed to extract_xvector_worker.py

set -e

# Parse arguments
GPU_LIST_STR=$1
PROCS_PER_GPU=$2
DATASET=$3
shift 3
EXTRA_ARGS="$@"

# Validate arguments
if [ -z "$GPU_LIST_STR" ] || [ -z "$PROCS_PER_GPU" ] || [ -z "$DATASET" ]; then
    echo "Usage: $0 <gpu_list> <procs_per_gpu> <dataset> [extra_args...]"
    echo "Example: $0 \"0,1,2,3\" 2 AISHELL-2"
    exit 1
fi

# Parse GPU list
IFS=',' read -ra GPU_LIST <<< "$GPU_LIST_STR"
NUM_GPUS=${#GPU_LIST[@]}
NUM_WORKERS=$((NUM_GPUS * PROCS_PER_GPU))

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Project root
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Log directory
# Keep logs next to this module (utils/Speaker_Embedding/logs/...) instead of repo root.
LOG_DIR="${SCRIPT_DIR}/logs/xvector"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Parallel X-Vector Extraction"
echo "========================================"
echo "GPU list: ${GPU_LIST[@]} (${NUM_GPUS} GPUs)"
echo "Processes per GPU: ${PROCS_PER_GPU}"
echo "Total workers: ${NUM_WORKERS}"
echo "Dataset: ${DATASET}"
echo "Extra args: ${EXTRA_ARGS}"
echo "Log directory: ${LOG_DIR}"
echo "========================================"

# Array to store PIDs
PIDS=()

# Launch workers
for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
    GPU_IDX=$((WORKER_ID / PROCS_PER_GPU))
    GPU_ID=${GPU_LIST[$GPU_IDX]}

    echo "Launching worker $WORKER_ID on GPU $GPU_ID"

    CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n 3D-Speaker python "${SCRIPT_DIR}/extract_xvector_worker.py" \
        --num-workers $NUM_WORKERS \
        --worker-id $WORKER_ID \
        --dataset $DATASET \
        --project-root "${PROJECT_ROOT}" \
        --log-dir "${LOG_DIR}" \
        $EXTRA_ARGS &

    PIDS+=($!)
done

echo "All ${NUM_WORKERS} workers launched. Waiting for completion..."
echo ""
echo "To monitor progress, check logs:"
echo "  tail -f ${LOG_DIR}/xvector_worker_*.log"
echo ""
echo "To kill the worker processes, run:"
echo "  pkill -f extract_xvector_worker.py"

# Wait for all workers
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if ! wait $PID; then
        echo "Worker $i (PID $PID) failed"
        FAILED=1
    fi
done

if [ $FAILED -eq 0 ]; then
    echo "========================================"
    echo "All workers completed successfully!"
    echo "========================================"
else
    echo "========================================"
    echo "Some workers failed. Check logs in ${LOG_DIR}"
    echo "========================================"
    exit 1
fi
