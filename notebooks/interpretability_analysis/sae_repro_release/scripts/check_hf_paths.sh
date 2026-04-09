#!/bin/bash
set -euo pipefail

METHOD="${1:-batch_topk}"
if [[ "$METHOD" != "batch_topk" && "$METHOD" != "sae_l1_penalty" ]]; then
  echo "Usage: $0 [batch_topk|sae_l1_penalty]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_resolve_paths.sh"
resolve_sae_paths "$METHOD"

echo "Method: $METHOD"
echo "EVA_REPO_ROOT: $EVA_REPO_ROOT"
echo "HF_MODEL_ROOT: $HF_MODEL_ROOT"
echo "HF_DATA_ROOT: $HF_DATA_ROOT"
echo "SAE_CKPT_DIR: $SAE_CKPT_DIR"
echo "HF_DATA_FASTA: $HF_DATA_FASTA"
echo "OK: path validation passed for $METHOD"
