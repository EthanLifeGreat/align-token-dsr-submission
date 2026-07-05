#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DSR="${PROJECT_DSR:-$(cd "${ROOT_DIR}/../../../../.." && pwd)}"
OUTPUT_DIR="${HIFIGAN_PRETRAIN_DIR:-${PROJECT_DSR}/results/vocoder_training/hifigan/bshall_legacy_matched/pretrain_aishell2_95_5}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${HIFIGAN_LD_PRELOAD:-}" ]]; then
  export LD_PRELOAD="${HIFIGAN_LD_PRELOAD}"
fi
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${HIFIGAN_CONDA_ENV:-hifigan}"

CUDA_VISIBLE_DEVICES="${HIFIGAN_GPUS:-1,2}" \
  "${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output \
  python train.py \
    --datasets \
      "${PROJECT_DSR}/data/AISHELL-2-95_5/meta.csv:${PROJECT_DSR}/data/AISHELL-2-95_5/wav" \
    --checkpoint_dir "${OUTPUT_DIR}" \
    --input_mel_source wav \
    --batch_size "${HIFIGAN_BATCH_SIZE:-8}" \
    --max_steps "${HIFIGAN_PRETRAIN_MAX_STEPS:-100000}" \
    --validation_interval "${HIFIGAN_VALIDATION_INTERVAL:-5000}" \
    --validation_subset "${HIFIGAN_VALIDATION_SUBSET:-1000}" \
    --early_stopping_patience "${HIFIGAN_EARLY_STOPPING_PATIENCE:-5}" \
    --early_stopping_min_delta "${HIFIGAN_EARLY_STOPPING_MIN_DELTA:-0.001}" \
    --early_stopping_min_steps "${HIFIGAN_EARLY_STOPPING_MIN_STEPS:-30000}" \
    --checkpoint_interval "${HIFIGAN_CHECKPOINT_INTERVAL:-5000}"
