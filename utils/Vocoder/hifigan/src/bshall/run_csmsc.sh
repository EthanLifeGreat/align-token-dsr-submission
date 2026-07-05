#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DSR="${PROJECT_DSR:-$(cd "${ROOT_DIR}/../../../../.." && pwd)}"
PRETRAIN_DIR="${HIFIGAN_PRETRAIN_DIR:-${PROJECT_DSR}/results/vocoder_training/hifigan/bshall_legacy_matched/pretrain_aishell2_95_5}"
OUTPUT_DIR="${HIFIGAN_FINETUNE_DIR:-${PROJECT_DSR}/results/vocoder_training/hifigan/bshall_legacy_matched/finetune_csmsc}"
PRETRAINED="${HIFIGAN_PRETRAIN_CHECKPOINT:-${PRETRAIN_DIR}/model-best.pt}"

if [[ ! -f "${PRETRAINED}" ]]; then
  echo "Missing AISHELL2 pretrained checkpoint: ${PRETRAINED}" >&2
  exit 1
fi

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
      "${PROJECT_DSR}/data/CSMSC/meta.csv:${PROJECT_DSR}/data/CSMSC/wav" \
    --checkpoint_dir "${OUTPUT_DIR}" \
    --resume "${PRETRAINED}" \
    --reset_optim \
    --input_mel_source wav \
    --disable_aug \
    --batch_size "${HIFIGAN_FINETUNE_BATCH_SIZE:-8}" \
    --learning_rate "${HIFIGAN_FINETUNE_LR:-0.0001}" \
    --max_steps "${HIFIGAN_FINETUNE_MAX_STEPS:-40000}" \
    --validation_interval "${HIFIGAN_FINETUNE_VALIDATION_INTERVAL:-1000}" \
    --validation_subset "${HIFIGAN_FINETUNE_VALIDATION_SUBSET:-500}" \
    --early_stopping_patience "${HIFIGAN_FINETUNE_PATIENCE:-5}" \
    --early_stopping_min_delta "${HIFIGAN_FINETUNE_MIN_DELTA:-0.001}" \
    --early_stopping_min_steps "${HIFIGAN_FINETUNE_MIN_STEPS:-10000}" \
    --checkpoint_interval "${HIFIGAN_FINETUNE_CHECKPOINT_INTERVAL:-5000}"
