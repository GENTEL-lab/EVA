#!/bin/bash
# RNA sequence generation launch script
# Usage: ./generate.sh <config.yaml>
# Example: ./generate.sh config_piRNA.yaml

set -e

# ===== Configuration =====
CONTAINER_NAME="eva"
PYTHON_PATH="/composer-python/python"
HOST_BASE="/path/to/your/host/project"  # Your host machine project path
CONTAINER_BASE="/eva"
GENERATE_SCRIPT="${CONTAINER_BASE}/tools/generate.py"

# ===== Argument Check =====
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [extra args...]"
    echo "Example: $0 config_piRNA.yaml"
    echo "      $0 config_piRNA.yaml --task h_sapiens_piRNA"
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

# Host path -> Container path
CONTAINER_CONFIG="${CONFIG_PATH/${HOST_BASE}/${CONTAINER_BASE}}"

echo "Config file: $CONFIG_PATH"
echo "Container path: $CONTAINER_CONFIG"
echo "Container name: $CONTAINER_NAME"
echo "---"

docker exec -it "$CONTAINER_NAME" "$PYTHON_PATH" "$GENERATE_SCRIPT" --config "$CONTAINER_CONFIG" "$@"
