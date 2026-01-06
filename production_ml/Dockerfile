# Production-ready Dockerfile with proper non-root user setup
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install all dependencies to /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install packages into the virtual environment
RUN pip install --no-cache-dir \
    --timeout=300 \
    --retries=5 \
    Flask==3.0.0 \
    gunicorn==21.2.0 \
    prometheus-client==0.19.0 \
    python-dotenv==1.0.0 \
    requests==2.31.0 \
    Pillow==10.1.0 && \
    pip install --no-cache-dir \
    --timeout=300 \
    --retries=5 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.0 \
    torchvision==0.16.0 && \
    pip install --no-cache-dir \
    --timeout=300 \
    --retries=5 \
    mlflow==2.9.2

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1000 -m -d /home/appuser appuser

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appuser src/ ./src/

# Make sure the venv is in PATH
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=src/main.py

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "src.main:app"]