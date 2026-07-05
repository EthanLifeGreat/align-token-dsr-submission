#!/bin/bash
# Extract Mel Spectrograms in parallel across multiple GPUs
#
# Usage:
#   ./extract_mel_parallel.sh <gpu_list> <procs_per_gpu> <dataset> [extra_args...]
#
# Example:
#   ./extract_mel_parallel.sh "0,1,2,3" 2 LibriTTS
#   ./extract_mel_parallel.sh "0,1" 1 MAGICDATA
#
# Args:
#   gpu_list: Comma-separated GPU IDs (e.g., "0,1,2,3")
#   procs_per_gpu: Number of processes per GPU
#   dataset: Dataset name (e.g., LibriTTS, MAGICDATA)
#   extra_args: Additional arguments passed to extract_mel_worker.py

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
    echo "Example: $0 \"0,1,2,3\" 2 LibriTTS"
    echo "Example: $0 \"0,1\" 1 MAGICDATA"
    exit 1
fi

# Parse GPU list
IFS=',' read -ra GPU_LIST <<< "$GPU_LIST_STR"
NUM_GPUS=${#GPU_LIST[@]}
NUM_WORKERS=$((NUM_GPUS * PROCS_PER_GPU))

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${MEL_CONDA_ENV:-dsr}"

# Create logs directory
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Parallel Mel Spectrogram Extraction"
echo "========================================"
echo "GPU list: ${GPU_LIST[@]} (${NUM_GPUS} GPUs)"
echo "Processes per GPU: ${PROCS_PER_GPU}"
echo "Total workers: ${NUM_WORKERS}"
echo "Dataset: ${DATASET}"
echo "Extra args: ${EXTRA_ARGS}"
echo "========================================"

# Array to store PIDs
PIDS=()

# Launch workers
for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
    GPU_IDX=$((WORKER_ID / PROCS_PER_GPU))
    GPU_ID=${GPU_LIST[$GPU_IDX]}

    LOG_FILE="${LOG_DIR}/worker_${WORKER_ID}.log"

    echo "Launching worker $WORKER_ID on GPU $GPU_ID (log: $LOG_FILE)"

    CUDA_VISIBLE_DEVICES=$GPU_ID "${CONDA_BIN}" run -n "${CONDA_ENV}" python "${SCRIPT_DIR}/extract_mel_worker.py" \
        --num-workers $NUM_WORKERS \
        --worker-id $WORKER_ID \
        --dataset $DATASET \
        $EXTRA_ARGS > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo "All ${NUM_WORKERS} workers launched. Waiting for completion..."

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
    echo "Logs saved to: ${LOG_DIR}"
    echo "========================================"
else
    echo "========================================"
    echo "Some workers failed. Check logs in: ${LOG_DIR}"
    echo "========================================"
    exit 1
fi
