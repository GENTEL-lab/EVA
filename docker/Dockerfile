# RNAVerse Docker Image
# Base: PyTorch Official Image (CUDA 12.4 + Python 3.11 + PyTorch 2.5.1)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime-ubuntu22.04

# Install flash-attn (requires special build flags)
# Set TORCH_CUDA_ARCH_LIST to skip CUDA detection during pip install
ENV TORCH_CUDA_ARCH_LIST=""
RUN MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation flash-attn==2.6.3

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy project code
COPY eva/ /eva/eva/
COPY tools/ /eva/tools/
COPY config/ /eva/config/
COPY data/test_data/ /eva/data/test_data/
COPY USAGE.md /eva/

# Create output directory
RUN mkdir -p /eva/data/output/scores

# Checkpoint should be mounted at runtime:
#   docker run -v /path/to/checkpoint:/eva/checkpoint ...

WORKDIR /eva
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["sleep", "infinity"]
