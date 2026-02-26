# =============================================================================
# Multi-stage Dockerfile for Enterprise LLM RAG Platform
# Optimized for GCP Cloud Run (serverless) and GKE
# No GPU required (Gemini Pro handles inference via API)
# =============================================================================

# ── Stage 1: Build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Production image ─────────────────────────────────────────────────
FROM python:3.11-slim AS production

# Security: run as non-root
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

# Install runtime-only system deps
# curl is required for the HEALTHCHECK command
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application source
COPY src/ ./src/

# Ensure scripts in .local are in PATH
ENV PATH=/root/.local/bin:$PATH

# ── HuggingFace model cache ───────────────────────────────────────────────────
# Models will be downloaded on first startup and cached in /tmp or a mounted volume.
# Pre-download in Docker build to avoid cold-start latency:
#   (comment out if you want a leaner image; model downloads ~500MB for bge-small)
ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# ── Runtime config ─────────────────────────────────────────────────────────────
# Cloud Run sets PORT automatically; default 8080
ENV PORT=8080
ENV ENV=production

# Switch to non-root (after model download which needs write access)
RUN chown -R appuser:appuser /app
USER appuser

# Health check (matches K8s readiness probe path)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

# Start FastAPI via uvicorn
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --log-level info"]
