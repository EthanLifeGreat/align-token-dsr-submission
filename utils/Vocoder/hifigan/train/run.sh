#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIFIGAN_ROOT="${HIFIGAN_ROOT:-${SCRIPT_DIR}/../src/bshall}"
STAGE="${1:-all}"

case "${STAGE}" in
  pretrain)
    exec "${HIFIGAN_ROOT}/run_aishell2.sh"
    ;;
  finetune)
    exec "${HIFIGAN_ROOT}/run_csmsc.sh"
    ;;
  all)
    exec "${HIFIGAN_ROOT}/run_all.sh"
    ;;
  *)
    echo "Usage: $0 {pretrain|finetune|all}" >&2
    exit 2
    ;;
esac
