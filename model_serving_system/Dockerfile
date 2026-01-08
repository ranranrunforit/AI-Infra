# Multi-stage build for optimized image size

# Stage 1: Builder
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights AS ROOT
RUN python -c "import torch; from torchvision.models import resnet50, ResNet50_Weights; print('Downloading ResNet50 model weights...'); model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); print('Model downloaded successfully!')"

# Verify the cache was created
RUN echo "=== Verifying model cache ===" && \
    ls -laR /root/.cache/ && \
    echo "=== Cache verification complete ==="

# Stage 2: Runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# Copy model cache to /app/.cache (accessible to appuser)
COPY --from=builder --chown=appuser:appuser /root/.cache /app/.cache

# Verify the cache was copied
RUN echo "=== Verifying cache in runtime stage ===" && \
    ls -laR /app/.cache/ && \
    echo "=== Runtime cache verification complete ===" && \
    stat /app/.cache && \
    echo "=== Ownership check complete ==="

# Copy application code
COPY --chown=appuser:appuser src/ ./src/

# Set environment variables - Point to /app/.cache instead of /home/appuser/.cache
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DEVICE=cpu \
    LOG_LEVEL=INFO \
    HOME=/home/appuser \
    TORCH_HOME=/app/.cache/torch \
    XDG_CACHE_HOME=/app/.cache

# Switch to non-root user
USER appuser

# Verify we can access the cache as appuser
RUN echo "=== Testing cache access as appuser ===" && \
    ls -la /app/.cache && \
    echo "=== Cache access test complete ==="

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

# Run the application
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]