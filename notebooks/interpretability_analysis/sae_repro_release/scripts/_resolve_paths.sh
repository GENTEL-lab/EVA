#!/bin/bash

# Shared resolver for SAE release scripts.
# Exports:
#   EVA_REPO_ROOT HF_MODEL_ROOT HF_DATA_ROOT HF_DATA_FASTA SAE_CKPT_DIR SAE_DATA_FASTA

resolve_sae_paths() {
  local method="${1:-batch_topk}"
  if [[ "$method" != "batch_topk" && "$method" != "sae_l1_penalty" ]]; then
    echo "ERROR: method must be batch_topk or sae_l1_penalty, got: $method" >&2
    return 1
  fi

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  export EVA_REPO_ROOT="${EVA_REPO_ROOT:-$(cd "$script_dir/../../../.." && pwd)}"
  export HF_MODEL_ROOT="${HF_MODEL_ROOT:-$EVA_REPO_ROOT/checkpoint}"
  export HF_DATA_ROOT="${HF_DATA_ROOT:-$EVA_REPO_ROOT/data/openrna/OpenRNA-v1-114M}"

  if [[ "$method" == "batch_topk" ]]; then
    export SAE_CKPT_DIR="${SAE_CKPT_DIR:-$HF_MODEL_ROOT/EVA_1.4B_CLM}"
  else
    export SAE_CKPT_DIR="${SAE_CKPT_DIR:-$HF_MODEL_ROOT/EVA_145M}"
  fi

  if [[ -z "${HF_DATA_FASTA:-}" ]]; then
    if [[ -d "$HF_DATA_ROOT" ]]; then
      local best=""
      local best_size=-1
      local f size
      while IFS= read -r -d '' f; do
        size=$(stat -c%s "$f" 2>/dev/null || echo 0)
        if [[ "$size" -gt "$best_size" ]]; then
          best="$f"
          best_size="$size"
        fi
      done < <(find "$HF_DATA_ROOT" -type f \( -name "*.fa" -o -name "*.fasta" \) -print0)
      if [[ -n "$best" ]]; then
        export HF_DATA_FASTA="$best"
      fi
    fi
  fi

  if [[ -z "${HF_DATA_FASTA:-}" ]]; then
    echo "ERROR: HF_DATA_FASTA is not set and no .fa/.fasta was found under HF_DATA_ROOT=$HF_DATA_ROOT" >&2
    echo "Set HF_DATA_FASTA directly or provide a valid HF_DATA_ROOT." >&2
    return 1
  fi
  if [[ ! -f "$HF_DATA_FASTA" ]]; then
    echo "ERROR: HF_DATA_FASTA not found: $HF_DATA_FASTA" >&2
    return 1
  fi
  if [[ ! -f "$SAE_CKPT_DIR/config.json" || ! -f "$SAE_CKPT_DIR/model_weights.pt" ]]; then
    echo "ERROR: Invalid SAE_CKPT_DIR (missing config.json or model_weights.pt): $SAE_CKPT_DIR" >&2
    return 1
  fi

  export SAE_DATA_FASTA="$HF_DATA_FASTA"
}
