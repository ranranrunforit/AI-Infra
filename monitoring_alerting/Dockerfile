FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    flask==2.3.0 \
    prometheus-client==0.17.0 \
    psutil==5.9.5 \
    numpy==1.24.3 \
    scipy==1.10.1

# Create log directory
RUN mkdir -p /var/log/app

# Copy application code
COPY src/instrumentation.py /app/
COPY src/app.py /app/

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]