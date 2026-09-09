# RNAVerse Docker Image
# Base: official PyTorch image (CUDA 12.4, Python 3.11, PyTorch 2.5.1).
# The 'devel' variant provides nvcc and CUDA headers, required to build
# grouped-gemm / stanford-stk / MegaBlocks CUDA extensions from source.
# EVA attention uses torch SDPA; the separate flash-attn package is not imported.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Optional package mirror; versions and algorithms remain pinned below.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_DEFAULT_TIMEOUT=60

# CUDA stubs must not leak into compile-time -lcuda resolution (Triton links
# its cuda_utils.so against libcuda); the real driver is injected at runtime
# by nvidia-container-toolkit (Docker) or --nv (Singularity/Apptainer).
ENV LIBRARY_PATH=""

# Deterministic GPU targets: Tesla T4 (7.5), Ampere/Ada via 8.0 binary
# compatibility, Hopper (9.0). Skips GPU detection, which would otherwise
# fail during docker build (no GPU available).
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;9.0"

# Build tooling, pinned for reproducible source builds below.
# Keep the established build-tool versions for the compiled MoE packages.
RUN pip install --no-cache-dir \
      setuptools==69.5.1 \
      wheel==0.45.1 \
      packaging==23.2 \
      "numpy==2.0.2" \
      triton==3.1.0 \
      ninja==1.11.1

# Keep git available for source-package metadata and full-source workflows.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# CUDA-extension packages must build against the preinstalled torch: install
# with --no-build-isolation so their setup.py can see it (with build
# isolation the isolated env has no torch and every build fails).
# Pre-pinning triton/numpy/packaging above keeps dependency resolution
# deterministic (megablocks only constrains triton>=2.1.0 and would
# otherwise pull a newer triton than the released environment uses).
# Build only the CUDA extensions used by the released model. No attention
# backend or optimization objective is changed when flash-attn is absent.
RUN MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation \
      grouped-gemm==0.1.6 \
      stanford-stk==0.7.1 \
      megablocks==0.7.0

# Never ship driver-library placeholders: images produced via `docker
# commit` of running GPU containers contain 0-byte libcuda.so.* files that
# break Triton's link step under Singularity --nv. Harmless on clean bases;
# the runtime driver is always mounted over these paths.
RUN rm -f /usr/lib/x86_64-linux-gnu/libcuda.so* \
          /usr/lib/x86_64-linux-gnu/libcudadebugger.so* \
          /usr/lib64/libcuda.so* \
          /usr/lib64/libcudadebugger.so* 2>/dev/null || true; \
    ldconfig 2>/dev/null || true

# Remaining dependencies (wheels / pure Python).
COPY scripts/docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Entrypoint: detects the real libcuda at runtime (Docker --gpus or
# Singularity --nv) and exports TRITON_LIBCUDA_PATH / TRITON_CACHE_DIR.
COPY scripts/docker/entrypoint.sh /usr/local/bin/eva-entrypoint.sh
RUN chmod +x /usr/local/bin/eva-entrypoint.sh

# Project code
COPY eva/ /eva/eva/
COPY tools/ /eva/tools/
COPY training/ /eva/training/
COPY examples/ /eva/examples/
COPY scripts/ /eva/scripts/
COPY tests/ /eva/tests/
COPY docs/ /eva/docs/
COPY .github/ /eva/.github/
COPY pyproject.toml README.md LICENSE CHANGELOG.md CITATION.cff CONTRIBUTING.md /eva/
RUN pip install --no-cache-dir --no-deps -e /eva

# Create output directory
RUN mkdir -p /eva/examples/data/output/scores

# Checkpoint should be mounted at runtime:
#   docker run -v /path/to/checkpoint:/eva/checkpoint ...

WORKDIR /eva
ENTRYPOINT ["/usr/local/bin/eva-entrypoint.sh"]
CMD ["sleep", "infinity"]

# Image provenance, overridable at build time:
#   docker build --build-arg EVA_VERSION=1.1.1 \
#                --build-arg EVA_REVISION=$(git rev-parse --short HEAD) ...
ARG EVA_VERSION=dev
ARG EVA_REVISION=unknown
LABEL org.opencontainers.image.version=$EVA_VERSION \
      org.opencontainers.image.revision=$EVA_REVISION \
      org.opencontainers.image.source="https://github.com/GENTEL-Lab/EVA"
