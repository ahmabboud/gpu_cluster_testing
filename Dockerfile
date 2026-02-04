# Dockerfile for GPU Cluster Acceptance Testing
#
# Base: NVIDIA PyTorch 24.07 (CUDA 12.5, Python 3.10)
# Purpose: Zero-dependency distributed training for cluster validation
#

FROM nvcr.io/nvidia/pytorch:24.07-py3

# Metadata
LABEL maintainer="Nebius Infrastructure Engineering"
LABEL description="GPU Cluster Acceptance Testing Tool"
LABEL version="1.0"

# Set working directory
WORKDIR /workspace

# Install system utilities and diagnostic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    iputils-ping \
    net-tools \
    dnsutils \
    curl \
    wget \
    vim \
    git \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Install Python diagnostic packages
RUN pip install --no-cache-dir \
    psutil \
    gpustat \
    torchvision

# Build NCCL tests for direct bandwidth/latency measurements
# This allows running focused NCCL tests alongside full training tests
RUN git clone https://github.com/NVIDIA/nccl-tests.git /workspace/nccl-tests && \
    cd /workspace/nccl-tests && \
    make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda && \
    echo "NCCL tests built successfully at /workspace/nccl-tests/build/"

# Create data directory for optional real datasets
RUN mkdir -p /workspace/data

# Copy application code
COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/

# Make entrypoint executable (if not already)
RUN chmod +x /workspace/scripts/entrypoint.sh

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Configure NCCL for verbose logging (can be overridden at runtime)
ENV NCCL_DEBUG=INFO
ENV NCCL_DEBUG_SUBSYS=ALL

# Set entrypoint
ENTRYPOINT ["/workspace/scripts/entrypoint.sh"]

# Default command: run ResNet-50 with default parameters
CMD ["--model", "resnet50", "--batch-size", "32"]
