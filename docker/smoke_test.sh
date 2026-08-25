#!/bin/bash
# Release smoke test for the EVA container: verify that Triton can build and
# load its CUDA helper library in BOTH container runtimes.
#
# Docker alone is NOT sufficient: --gpus mounts the real driver over any
# broken libcuda placeholders in the image, which hides exactly the class of
# bugs that breaks Singularity users (see eva/_runtime_env.py for details).
# Always run the Singularity variant before publishing an image.
#
# Usage:
#   docker/smoke_test.sh docker [image]          # default: eva:latest
#   docker/smoke_test.sh singularity [sif]       # default: eva_latest.sif
#
# Exit code is 0 only if the Triton driver probe succeeds.

set -euo pipefail

mode="${1:-}"
shift || true

PROBE='import torch; from triton.runtime import driver; import eva; \
print("torch:", torch.__version__); \
print("eva:", eva.__version__); \
print("cuda available:", torch.cuda.is_available()); \
print("triton driver:", driver.active); \
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"'

case "$mode" in
    docker)
        image="${1:-eva:latest}"
        echo "== Docker smoke test: $image =="
        docker run --rm --gpus all "$image" python -c "$PROBE"
        docker run --rm --gpus all "$image" eva-generate --help >/dev/null
        ;;
    singularity)
        sif="${1:-eva_latest.sif}"
        if ! command -v singularity >/dev/null 2>&1 && command -v apptainer >/dev/null 2>&1; then
            singularity() { apptainer "$@"; }
        fi
        echo "== Singularity smoke test: $sif =="
        # --cleanenv keeps host environment variables from leaking in, which
        # mirrors how HPC users invoke the container.
        singularity exec --cleanenv --nv "$sif" python -c "$PROBE"
        singularity exec --cleanenv --nv "$sif" eva-generate --help >/dev/null
        ;;
    *)
        echo "Usage: $0 {docker [image] | singularity [sif]}" >&2
        echo "  docker        test image 'eva:latest' (override: $0 docker my-image:tag)" >&2
        echo "  singularity   test 'eva_latest.sif'   (override: $0 singularity my.sif)" >&2
        exit 2
        ;;
esac

echo "== smoke test passed ($mode) =="
