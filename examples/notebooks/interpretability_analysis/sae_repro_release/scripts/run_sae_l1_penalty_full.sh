#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_ensure_eva_container.sh"
sae_require_eva_container "${BASH_SOURCE[0]}" "$@"

PYTHON_BIN="${PYTHON_BIN:-python3}"
source "$SCRIPT_DIR/_resolve_paths.sh"
resolve_sae_paths "sae_l1_penalty"

export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29501}"

echo "Resolved checkpoint: $SAE_CKPT_DIR"
echo "Resolved FASTA: $SAE_DATA_FASTA"

"$PYTHON_BIN" "$SCRIPT_DIR/run_training.py" \
  --config "$SCRIPT_DIR/../configs/config_sae_l1_penalty_full.yaml" \
  --mode sae_l1_penalty
