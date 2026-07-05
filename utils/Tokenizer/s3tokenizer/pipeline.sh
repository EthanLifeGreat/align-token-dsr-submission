#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PIPELINE_PY="${SCRIPT_DIR}/pipeline.py"

if [[ ! -f "${PIPELINE_PY}" ]]; then
  echo "pipeline.py not found: ${PIPELINE_PY}" >&2
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "python3/python not found in PATH" >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${PIPELINE_PY}" --project-root "${PROJECT_ROOT}" "$@"
