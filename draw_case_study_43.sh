#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
if [ "$(basename "${ROOT_DIR}")" = "scripts" ]; then
  ROOT_DIR="$(dirname "${ROOT_DIR}")"
fi
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Python not found. Set PYTHON=/path/to/python." >&2
    exit 127
  fi
fi

exec "${PYTHON_BIN}" scripts/run_batches.py draw-case-study-43 "$@"