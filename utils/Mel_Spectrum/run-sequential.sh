#!/bin/bash
# Extract Mel Spectrograms for a specific dataset
# Usage: ./run.sh

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET="${1:-${DATASET:-CSMSC}}"

# Build paths
DATA_DIR="${PROJECT_ROOT}/data/${DATASET}"
MEL_OUTPUT_DIR="${DATA_DIR}/mel"

echo "========================================"
echo "Mel Spectrogram Extraction"
echo "========================================"
echo "Dataset: ${DATASET}"
echo "Data directory: ${DATA_DIR}"
echo "Mel output directory: ${MEL_OUTPUT_DIR}"
echo "========================================"

# Set PYTHONPATH
export PYTHONPATH=${PROJECT_ROOT}

# Run extraction
python "${SCRIPT_DIR}/extract_mel.py" \
    --dataset "${DATASET}" \
    --project-root "${PROJECT_ROOT}" \
    --mel_output_dir "${MEL_OUTPUT_DIR}"

echo "========================================"
echo "Extraction completed!"
echo "========================================"
