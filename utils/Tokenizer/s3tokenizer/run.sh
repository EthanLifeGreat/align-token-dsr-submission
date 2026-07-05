#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=("AISHELL-2-95_5" "CSMSC")
fi

exec "${SCRIPT_DIR}/pipeline.sh" \
 --model speech_tokenizer_v2_25hz \
 --datasets "${DATASETS[@]}" \
 --gpus "${S3TOKENIZER_GPUS:-0,1,2,3}"
