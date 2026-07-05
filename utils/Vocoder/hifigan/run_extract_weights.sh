#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CHECKPOINT_DIR="${HIFIGAN_CHECKPOINT_DIR:-${ROOT_DIR}/results/vocoder_training/hifigan/bshall_legacy_matched/finetune_csmsc}"
INPUT_CHECKPOINT="${HIFIGAN_INPUT_CHECKPOINT:-${CHECKPOINT_DIR}/model-best.pt}"
OUTPUT_WEIGHTS="${HIFIGAN_OUTPUT_WEIGHTS:-${SCRIPT_DIR}/ckpt/generator-csmsc.pt}"

if [[ ! -f "${INPUT_CHECKPOINT}" ]]; then
    echo "Input checkpoint not found: ${INPUT_CHECKPOINT}" >&2
    exit 1
fi

echo "Extracting best checkpoint: ${INPUT_CHECKPOINT}"
python "${SCRIPT_DIR}/extract_weights.py" "${INPUT_CHECKPOINT}" --output "${OUTPUT_WEIGHTS}"
