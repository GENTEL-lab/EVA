#!/bin/bash
# RNA sequence scoring launch script
# Usage: ./predict.sh <config.yaml>
# Example: ./predict.sh config_score_mRNA.yaml

set -e

# ===== Configuration =====
CONTAINER_NAME="eva"
PYTHON_PATH="/composer-python/python"
HOST_BASE="/path/to/your/host/project"  # Your host machine project path
CONTAINER_BASE="/eva"
PREDICT_SCRIPT="${CONTAINER_BASE}/tools/predict.py"

# ===== Argument Check =====
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [extra args...]"
    echo "Example: $0 config_score_mRNA.yaml"
    echo "      $0 config_score_mRNA.yaml --device cuda:1"
    exit 1
fi

CONFIG_PATH="$1"
shift

# Convert relative path to absolute path if needed
if [[ "$CONFIG_PATH" != /* ]]; then
    CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
fi

# Check if file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: config file not found: $CONFIG_PATH"
    exit 1
fi

# Host path -> Container path
CONTAINER_CONFIG="${CONFIG_PATH/${HOST_BASE}/${CONTAINER_BASE}}"

echo "Config file: $CONFIG_PATH"
echo "Container path: $CONTAINER_CONFIG"
echo "Container name: $CONTAINER_NAME"
echo "---"

docker exec "$CONTAINER_NAME" "$PYTHON_PATH" "$PREDICT_SCRIPT" --config "$CONTAINER_CONFIG" "$@"
