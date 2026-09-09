#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/eva_prediction_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

cd "${SCRIPT_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/reproduce_prediction_figures.py" "$@"
