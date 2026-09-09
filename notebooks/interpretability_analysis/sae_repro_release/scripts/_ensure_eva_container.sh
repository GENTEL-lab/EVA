#!/bin/bash

sae_in_eva_container() {
  if [[ "${EVA_SAE_IN_CONTAINER:-}" == "1" ]]; then
    return 0
  fi

  local container_workdir="${CONTAINER_WORKDIR:-/eva}"
  [[ -f /.dockerenv && -d "$container_workdir/notebooks/interpretability_analysis/sae_repro_release" ]]
}

sae_require_eva_container() {
  local script_ref="$1"
  shift || true

  if [[ "${EVA_SAE_LOCAL:-0}" == "1" ]]; then
    return 0
  fi

  export CONTAINER_NAME="${CONTAINER_NAME:-eva-repro}"
  export CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/eva}"

  if sae_in_eva_container; then
    return 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker command not found. Start ${CONTAINER_NAME} and rerun this script." >&2
    exit 1
  fi

  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "ERROR: Docker container '${CONTAINER_NAME}' is not running." >&2
    echo "Start it first, then rerun this script." >&2
    exit 1
  fi

  local script_dir script_abs repo_root script_rel container_script
  script_dir="$(cd "$(dirname "$script_ref")" && pwd)"
  script_abs="$script_dir/$(basename "$script_ref")"
  repo_root="$(cd "$script_dir/../../../.." && pwd)"

  if [[ "$script_abs" != "$repo_root/"* ]]; then
    echo "ERROR: Cannot map script path into repo root: $script_abs" >&2
    exit 1
  fi

  script_rel="${script_abs#"$repo_root"/}"
  container_script="${CONTAINER_WORKDIR}/${script_rel}"

  local -a docker_cmd=(docker exec)
  if [[ -t 0 && -t 1 ]]; then
    docker_cmd+=(-it)
  else
    docker_cmd+=(-i)
  fi

  docker_cmd+=(
    -w "$CONTAINER_WORKDIR"
    -e "EVA_SAE_IN_CONTAINER=1"
    -e "EVA_REPO_ROOT=$CONTAINER_WORKDIR"
    -e "CONTAINER_NAME=$CONTAINER_NAME"
    -e "CONTAINER_WORKDIR=$CONTAINER_WORKDIR"
  )

  local passthrough_vars=(
    HF_MODEL_ROOT
    HF_DATA_ROOT
    HF_DATA_FASTA
    SAE_CKPT_DIR
    SAE_DATA_FASTA
    PYTHON_BIN
    MASTER_PORT
    SAE_OUTPUT_TAG
    SAE_OUTPUT_ROOT
    CUDA_VISIBLE_DEVICES
  )
  local var_name
  for var_name in "${passthrough_vars[@]}"; do
    if [[ -n "${!var_name:-}" ]]; then
      docker_cmd+=(-e "${var_name}=${!var_name}")
    fi
  done

  echo "[sae] Running inside container '${CONTAINER_NAME}': ${script_rel}"
  exec "${docker_cmd[@]}" "$CONTAINER_NAME" bash "$container_script" "$@"
}
