FROM python:3.11-slim

WORKDIR /mlflow

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    mlflow==2.8.1 \
    psycopg2-binary \
    boto3

# Expose MLflow port
EXPOSE 5000

# Default command (overridden by docker-compose)
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]