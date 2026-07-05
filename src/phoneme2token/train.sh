#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG_NAME="${1:-debug}"
GPUS="${GPUS:-0}"
CONDA_ENV="${CONDA_ENV:-dsr}"

bash "$PROJECT_ROOT/src/train_utils/train.sh" \
    --model phoneme2token \
    --model-config "$PROJECT_ROOT/src/phoneme2token/configs/$CONFIG_NAME/config.yaml" \
    --train-config "$PROJECT_ROOT/src/phoneme2token/configs/$CONFIG_NAME/train_config.yaml" \
    --gpus "$GPUS" \
    --conda-env "$CONDA_ENV"

