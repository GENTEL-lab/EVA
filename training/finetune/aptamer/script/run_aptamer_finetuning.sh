#!/usr/bin/env bash
# Single-GPU finetuning; host source is mounted at CONTAINER_WORKDIR.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST_PROJECT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/eva}"
CONTAINER_NAME="${CONTAINER_NAME:-eva-repro}"
GPU_ID=0
CONFIG_FILE=""
EXPERIMENT_NAME=""
FOREGROUND=0
usage() {
    printf '%s\n' 'Usage: run_aptamer_finetuning.sh [experiment-directory] [--config PATH] [--gpu CONTAINER_GPU_ID] [--container NAME] [--foreground]' 'RNA type is data_config.rna_type_token in YAML, not a CLI option.'
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|--gpu|--container)
            [[ $# -ge 2 && "$2" != --* ]] || { usage >&2; exit 2; }
            case "$1" in
                --config) CONFIG_FILE="$2";;
                --gpu) GPU_ID="$2";;
                --container) CONTAINER_NAME="$2";;
            esac
            shift 2;;
        --foreground) FOREGROUND=1; shift;;
        --help|-h) usage; exit 0;;
        -*) printf 'Unknown option: %s\n' "$1" >&2; exit 2;;
        *) [[ -z "$EXPERIMENT_NAME" ]] || { usage >&2; exit 2; }
           EXPERIMENT_NAME="$1"; shift;;
    esac
done
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { printf '%s\n' '--gpu must be a single nonnegative integer' >&2; exit 2; }
if [[ -z "$CONFIG_FILE" ]]; then
    case "$EXPERIMENT_NAME" in
        '') CONFIG_FILE=training/finetune/aptamer/script/experiment_config_template.yaml;;
        /*|results/*) CONFIG_FILE="${EXPERIMENT_NAME%/}/experiment_config.yaml";;
        *) CONFIG_FILE="results/aptamer_finetuning/$EXPERIMENT_NAME/experiment_config.yaml";;
    esac
fi
case "$CONFIG_FILE" in
    "$HOST_PROJECT_DIR"/*) CONFIG_FILE="$CONTAINER_WORKDIR/${CONFIG_FILE#"$HOST_PROJECT_DIR"/}";;
    /*) ;;
    *) CONFIG_FILE="$CONTAINER_WORKDIR/$CONFIG_FILE";;
esac
[[ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME")" == true ]] || { printf '%s\n' 'Container is not running' >&2; exit 1; }
docker exec "$CONTAINER_NAME" test -f "$CONFIG_FILE"
docker exec "$CONTAINER_NAME" test -f "$CONTAINER_WORKDIR/training/finetune/train_finetune.py"
cmd=(docker exec -w "$CONTAINER_WORKDIR" -e "CUDA_VISIBLE_DEVICES=$GPU_ID" "$CONTAINER_NAME" python training/finetune/train_finetune.py --config "$CONFIG_FILE")
if [[ "$FOREGROUND" == 1 ]]; then
    exec "${cmd[@]}"
fi
LOG_DIR="$HOST_PROJECT_DIR/results/logs/aptamer_finetuning"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/aptamer_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup "${cmd[@]}" > "$LOG_FILE" 2>&1 &
printf 'Launched PID %s; completion is not yet verified. Log: %s\n' "$!" "$LOG_FILE"
