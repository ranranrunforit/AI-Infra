FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl vim net-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

RUN mkdir -p /mnt/data /mnt/checkpoints

EXPOSE 6379 8265 10001

CMD ["ray", "start", "--head", "--port=6379", "--dashboard-host=0.0.0.0", "--dashboard-port=8265", "--block"]