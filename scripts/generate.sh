#!/bin/bash
# RNA sequence generation launch script for a running EVA Docker container.
# Usage: ./scripts/generate.sh <config.yaml> [extra args...]
# Example: ./scripts/generate.sh config/tools_config/config_clm_example.yaml

set -e

# ===== Configuration =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTAINER_NAME="${CONTAINER_NAME:-eva}"
PYTHON_PATH="${PYTHON_PATH:-python}"
HOST_BASE="${HOST_BASE:-${REPO_ROOT}}"
CONTAINER_BASE="${CONTAINER_BASE:-/eva}"
GENERATE_SCRIPT="${GENERATE_SCRIPT:-${CONTAINER_BASE}/tools/generate.py}"

# ===== Argument Check =====
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [extra args...]"
    echo "Example: $0 config/tools_config/config_clm_example.yaml"
    echo "      $0 config/tools_config/config_clm_example.yaml --task human_mRNA"
    exit 1
fi

CONFIG_PATH="$1"
shift  # Pass remaining arguments as extra parameters

# Convert relative path to absolute path if needed
if [[ "$CONFIG_PATH" != /* ]]; then
    CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
fi

# Check if file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: config file not found: $CONFIG_PATH"
    exit 1
fi

# Host path -> container path. HOST_BASE must be mounted at CONTAINER_BASE.
if [[ "$CONFIG_PATH" == "$HOST_BASE"/* ]]; then
    RELATIVE_CONFIG="${CONFIG_PATH#${HOST_BASE}/}"
    CONTAINER_CONFIG="${CONTAINER_BASE}/${RELATIVE_CONFIG}"
else
    CONTAINER_CONFIG="$CONFIG_PATH"
fi

echo "Config file: $CONFIG_PATH"
echo "Container path: $CONTAINER_CONFIG"
echo "Container name: $CONTAINER_NAME"
echo "---"

docker exec -it "$CONTAINER_NAME" "$PYTHON_PATH" "$GENERATE_SCRIPT" --config "$CONTAINER_CONFIG" "$@"
