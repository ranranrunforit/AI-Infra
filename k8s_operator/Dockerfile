FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy operator code
COPY src/ ./src/

# Create non-root user
RUN useradd -m -u 1000 operator && \
    chown -R operator:operator /app

USER operator

# Run the operator
CMD ["python", "-m", "src.operator.main"]
