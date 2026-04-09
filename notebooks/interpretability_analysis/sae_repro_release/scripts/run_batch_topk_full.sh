#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_resolve_paths.sh"
resolve_sae_paths "batch_topk"

export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "Resolved checkpoint: $SAE_CKPT_DIR"
echo "Resolved FASTA: $SAE_DATA_FASTA"

"$PYTHON_BIN" "$SCRIPT_DIR/run_training.py" \
  --config "$SCRIPT_DIR/../configs/config_batch_topk_full.yaml" \
  --mode batch_topk
