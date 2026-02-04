# Multi-stage Docker build for distributed training

# Stage 1: Build dependencies
FROM nvcr.io/nvidia/pytorch:23.12-py3 as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime image
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Set environment variables for NCCL optimization
ENV NCCL_DEBUG=INFO \
    NCCL_IB_DISABLE=0 \
    NCCL_NET_GDR_LEVEL=5 \
    NCCL_SOCKET_IFNAME=eth0 \
    NCCL_NSOCKS_PERTHREAD=4 \
    NCCL_BUFFSIZE=2097152 \
    OMP_NUM_THREADS=8 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    infiniband-diags \
    ibverbs-utils \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash ray && \
    mkdir -p /app /mnt/data /mnt/checkpoints && \
    chown -R ray:ray /app /mnt/data /mnt/checkpoints

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=ray:ray src/ ./src/
COPY --chown=ray:ray scripts/ ./scripts/
COPY --chown=ray:ray configs/ ./configs/

# Switch to non-root user
USER ray

# Expose Ray ports
EXPOSE 6379 8265 10001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import ray; ray.init(address='auto', ignore_reinit_error=True)" || exit 1

# Default command
CMD ["python", "-m", "src.training.distributed_trainer"]
