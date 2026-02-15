# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy operator code
COPY src/ ./src/

# Create non-root user
RUN useradd -m -u 1000 operator && \
    chown -R operator:operator /app

USER operator

# Expose ports
# 8080: Kopf healthz/liveness
# 9090: Prometheus metrics
EXPOSE 8080 9090

# Run the operator
CMD ["python", "-m", "src.operator.main"]
