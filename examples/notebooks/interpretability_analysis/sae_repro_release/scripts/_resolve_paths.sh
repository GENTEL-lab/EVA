#!/usr/bin/env bash
# Explicit scientific inputs: do not pick the largest FASTA or substitute weights.
resolve_sae_paths() {
  local method="${1:-batch_topk}"
  [[ "$method" == batch_topk || "$method" == sae_l1_penalty ]] || { printf '%s\n' 'Invalid SAE method' >&2; return 1; }
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export EVA_REPO_ROOT="${EVA_REPO_ROOT:-$(cd "$script_dir/../../../../.." && pwd)}"
  export HF_MODEL_ROOT="${HF_MODEL_ROOT:-$EVA_REPO_ROOT/checkpoint}"
  if [[ -z "${SAE_CKPT_DIR:-}" ]]; then
    if [[ "$method" == batch_topk ]]; then
      export SAE_CKPT_DIR="$HF_MODEL_ROOT/EVA_1.4B_CLM"
    else
      export SAE_CKPT_DIR="$HF_MODEL_ROOT/EVA_145M"
    fi
  fi
  export HF_DATA_FASTA="${HF_DATA_FASTA:-${SAE_DATA_FASTA:-}}"
  [[ -n "$HF_DATA_FASTA" && -f "$HF_DATA_FASTA" ]] || {
    printf '%s\n' 'Set HF_DATA_FASTA to the exact intended training split; automatic FASTA selection is disabled.' >&2
    return 1
  }
  for file in config.json tokenizer.json model_weights.pt; do
    [[ -f "$SAE_CKPT_DIR/$file" ]] || { printf 'Missing checkpoint file: %s/%s\n' "$SAE_CKPT_DIR" "$file" >&2; return 1; }
  done
  export HF_DATA_ROOT="${HF_DATA_ROOT:-$(dirname "$HF_DATA_FASTA")}"
  export SAE_DATA_FASTA="$HF_DATA_FASTA"
}
