#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_ensure_eva_container.sh"
sae_require_eva_container "${BASH_SOURCE[0]}" "$@"

bash "$SCRIPT_DIR/check_hf_paths.sh" batch_topk
bash "$SCRIPT_DIR/check_hf_paths.sh" sae_l1_penalty

bash "$SCRIPT_DIR/run_batch_topk_full.sh"
bash "$SCRIPT_DIR/run_sae_l1_penalty_full.sh"
