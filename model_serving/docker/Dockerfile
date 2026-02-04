# Multi-stage Dockerfile for High-Performance Model Serving
# Optimized for production deployment with TensorRT and vLLM support

# =============================================================================
# Stage 1: Base Image with CUDA and System Dependencies
# =============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# Metadata
LABEL maintainer="ai-infra-curriculum@joshua-ferguson.com"
LABEL description="High-Performance Model Serving with TensorRT and vLLM"
LABEL version="1.0.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    # Python dependencies
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    # CUDA and GPU tools
    libnvinfer8 \
    libnvinfer-plugin8 \
    libnvonnxparsers8 \
    libnvparsers8 \
    # System libraries
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    libssl-dev \
    libffi-dev \
    # Networking and monitoring
    net-tools \
    iproute2 \
    procps \
    htop \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel

# =============================================================================
# Stage 2: Builder - Install Python Dependencies
# =============================================================================
FROM base AS builder

# Set working directory for build
WORKDIR /build

# Copy requirements file
COPY requirements.txt .

# Install Python packages with optimizations
RUN pip install --user --no-warn-script-location \
    # Core dependencies
    -r requirements.txt \
    # Additional production dependencies
    gunicorn==21.2.0 \
    uvicorn[standard]==0.24.0 \
    # Monitoring and tracing
    prometheus-client==0.19.0 \
    opentelemetry-api==1.21.0 \
    opentelemetry-sdk==1.21.0 \
    opentelemetry-instrumentation-fastapi==0.42b0 \
    opentelemetry-exporter-jaeger==1.21.0 \
    opentelemetry-exporter-otlp==1.21.0 \
    # GPU acceleration
    nvidia-ml-py==12.535.133 \
    # Performance optimization
    orjson==3.9.10 \
    ujson==5.8.0 \
    msgpack==1.0.7

# =============================================================================
# Stage 3: TensorRT Installation
# =============================================================================
FROM base AS tensorrt-builder

# Install TensorRT from NVIDIA repositories
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11-dev \
    libnvinfer-dev=8.6.1.6-1+cuda12.0 \
    libnvonnxparsers-dev=8.6.1.6-1+cuda12.0 \
    libnvinfer-plugin-dev=8.6.1.6-1+cuda12.0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT Python bindings
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install TensorRT Python package
RUN pip install --user \
    tensorrt==8.6.1.post1 \
    nvidia-tensorrt==99.0.0 \
    pycuda==2022.2.2

# =============================================================================
# Stage 4: Final Production Image
# =============================================================================
FROM base AS production

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=tensorrt-builder /root/.local /home/appuser/.local

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/models \
    /app/engine_cache \
    /app/calibration_cache \
    /app/logs \
    /app/checkpoints \
    /app/profiles \
    && chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser scripts/ /app/scripts/
COPY --chown=appuser:appuser .env.example /app/.env

# Set environment variables for the application
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app:$PYTHONPATH \
    MODEL_CACHE_DIR=/app/models \
    TENSORRT_ENGINE_CACHE_DIR=/app/engine_cache \
    TENSORRT_CALIBRATION_CACHE=/app/calibration_cache \
    LOG_FILE=/app/logs/model-serving.log \
    CHECKPOINT_DIR=/app/checkpoints \
    PROFILE_OUTPUT_DIR=/app/profiles

# Expose ports
# 8000: Main API server
# 9090: Prometheus metrics
EXPOSE 8000 9090

# Switch to non-root user
USER appuser

# Health check configuration
# Check both the liveness endpoint and GPU availability
HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=5m \
    --retries=3 \
    CMD python -c "import requests; import sys; \
                   r = requests.get('http://localhost:8000/health/live', timeout=5); \
                   sys.exit(0 if r.status_code == 200 else 1)"

# Set default command (can be overridden)
CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info", \
     "--no-access-log"]

# =============================================================================
# Stage 5: Development Image (with additional tools)
# =============================================================================
FROM production AS development

USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    tmux \
    gdb \
    strace \
    tcpdump \
    ipython3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --user \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    pytest-cov==4.1.0 \
    pytest-mock==3.12.0 \
    black==23.12.1 \
    isort==5.13.2 \
    flake8==7.0.0 \
    mypy==1.8.0 \
    pylint==3.0.3 \
    ipdb==0.13.13 \
    memory-profiler==0.61.0 \
    line-profiler==4.1.1

# Enable debug mode
ENV DEBUG_MODE=true \
    PROFILE_REQUESTS=true \
    VERBOSE_ERROR_MESSAGES=true

USER appuser

# Override CMD for development with auto-reload
CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--log-level", "debug"]
