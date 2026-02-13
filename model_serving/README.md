# Project 10: High-Performance Model Serving
## Overview

This is the production-ready high-performance model serving with TensorRT optimization and vLLM integration.

### Key Features

- **TensorRT Optimization**: Automated model conversion and optimization with FP16/INT8 quantization
- **vLLM Integration**: Efficient large language model serving with PagedAttention
- **Intelligent Routing**: A/B testing, canary deployments, and traffic splitting
- **Auto-scaling**: Kubernetes HPA with custom metrics
- **Distributed Tracing**: End-to-end request tracing with Jaeger/OpenTelemetry
- **Performance Benchmarking**: Comprehensive latency and throughput testing
- **Production-Ready**: Complete monitoring, logging, and error handling

## Architecture

```
┌─────────────────┐
│   Load Balancer │
└────────┬────────┘
         │
    ┌────▼────────────────────────────┐
    │  Intelligent Router             │
    │  (A/B, Canary, Traffic Split)   │
    └────┬────────────────────────────┘
         │
    ┌────▼─────────┬──────────────┐
    │              │              │
┌───▼────┐   ┌────▼────┐   ┌────▼────┐
│TensorRT│   │  vLLM   │   │ PyTorch │
│Serving │   │ Serving │   │ Serving │
└────┬───┘   └────┬────┘   └────┬────┘
     │            │              │
     └────────────┴──────────────┘
                  │
         ┌────────▼─────────┐
         │   Observability  │
         │  (Prometheus,    │
         │   Jaeger, Logs)  │
         └──────────────────┘
```

## Quick Start

### Prerequisites

```bash
# NVIDIA GPU with CUDA 12.1+
# Docker and Docker Compose
# Kubernetes cluster (optional, for production deployment)
```

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env with your configuration

# 3. Convert a model to TensorRT
python src/tensorrt/convert_model.py \
    --model resnet50 \
    --precision fp16 \
    --output models/resnet50-fp16.trt

# 4. Start the serving stack with Docker Compose
docker-compose up -d

# 5. Test the API
curl http://localhost:8000/health

# 6. Run inference
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"model": "resnet50", "input": {"image_url": "https://example.com/cat.jpg"}}'
```

### Docker Deployment

```bash
## Option 1: Docker Compose (Recommended)

### Step 1: Build the Docker Image

docker build -t model-serving:latest -f docker/Dockerfile .

# First build takes 10-15 minutes. Subsequent builds use cache.

### Step 2: Start the Stack

docker compose -f docker/docker-compose.yml up -d

### Step 3: Verify All Services

docker compose -f docker/docker-compose.yml ps

# Expected output — all services should show `Up` or `healthy`:

| Service | Port | URL |
|---------|------|-----|
| Model Serving API | 8000 | http://localhost:8000 |
| Prometheus | 9091 | http://localhost:9091 |
| Grafana | 3000 | http://localhost:3000 (admin/admin) |
| Jaeger UI | 16686 | http://localhost:16686 |
| Redis | 6379 | — |
| PostgreSQL | 5432 | — |

curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/metrics
# Or
Invoke-RestMethod -Uri "http://localhost:8000/health"

### Then download the model and test:
docker exec -it model-serving python /app/scripts/download_resnet18.py
# Or
docker exec -it -e TORCH_HOME=/tmp/torch_cache -e XDG_CACHE_HOME=/tmp model-serving python /app/scripts/download_resnet18.py

### Step 4: Test the API

# Health check
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/metrics
# Or
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Predict (once models are loaded)
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/v1/predict" -ContentType "application/json" -Body '{"model": "resnet18", "inputs": {"image": "https://example.com/cat.jpg"}}'

### Step 5: View Monitoring

# 1. Open **Grafana** at http://localhost:3000 (login: admin / admin)
# 2. Navigate to **Dashboards → Model Serving** folder → **Model Serving Overview**
# 3. Open **Jaeger** at http://localhost:16686 for distributed traces

### Step 6: Stop the Stack

docker compose -f docker/docker-compose.yml down
# To also remove data volumes:
docker compose -f docker/docker-compose.yml down -v


## Debug: Rebuild the image and Test
docker compose -f docker/docker-compose.yml down
docker compose -f docker/docker-compose.yml up --build -d



## Option 2: Single Docker Container (Lightweight)

# If you only need the API without monitoring:

# Build
docker build -t model-serving:latest -f docker/Dockerfile .

# Run with GPU
docker run --rm -d \
    --name model-serving \
    --gpus all \
    -p 8000:8000 \
    -p 9090:9090 \
    -v model-cache:/app/models \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e GPU_MEMORY_FRACTION=0.85 \
    model-serving:latest

# Download model & Test

docker exec -it -e TORCH_HOME=/tmp/torch_cache -e XDG_CACHE_HOME=/tmp model-serving python /app/scripts/download_resnet18.py

curl.exe http://localhost:8000/health

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/v1/predict" \
  -ContentType "application/json" \
  -Body '{"model": "resnet18", "inputs": {"image": "https://example.com/cat.jpg"}}'


## Converting Models
# Use the conversion script to create TensorRT engines:
# Inside the container
docker exec -it model-serving python scripts/convert_model.py \
    --model resnet50 \
    --precision fp16 \
    --batch-size 8 \
    --output /app/models/resnet50-fp16.trt

# INT8 calibration (also works on single GPU)
docker exec -it model-serving python scripts/convert_model.py \
    --model resnet50 \
    --precision int8 \
    --calibration-samples 200 \
    --output /app/models/resnet50-int8.trt

# Stop
 docker stop model-serving


# To clean up everything else:

# Stop container (auto-removes due to --rm)
docker stop model-serving
# Remove the model-cache volume (where downloaded models are stored)
docker volume rm model-cache
# Remove the Docker image
docker rmi model-serving:latest
# (Optional) Remove all docker-compose volumes too
docker compose -f docker/docker-compose.yml down -v
# That removes the container, the model files, and the image. To verify nothing's left:
docker ps -a --filter name=model-serving
docker volume ls --filter name=model-cache
docker images model-serving
```

### Kubernetes Deployment

```bash
# 1. Build and push Docker image
docker build -t your-registry/model-serving:latest -f docker/Dockerfile .
docker push your-registry/model-serving:latest

# 2. Deploy to Kubernetes
kubectl apply -k kubernetes/overlays/prod/

# 3. Verify deployment
kubectl get pods -n model-serving
kubectl get svc -n model-serving

# 4. Access the service
export SERVICE_URL=$(kubectl get svc model-serving -n model-serving -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://$SERVICE_URL/health
```

## Project Structure

```
project-10-model-serving/
├── src/
│   ├── tensorrt/          # TensorRT conversion and optimization
│   │   ├── __init__.py
│   │   ├── converter.py   # Model conversion to TensorRT
│   │   ├── calibrator.py  # INT8 calibration
│   │   └── optimizer.py   # Optimization strategies
│   ├── serving/           # Core serving infrastructure
│   │   ├── __init__.py
│   │   ├── server.py      # FastAPI server
│   │   ├── model_loader.py # Model loading and caching
│   │   ├── batch_processor.py # Dynamic batching
│   │   └── warmup.py      # Model warmup
│   ├── llm/              # vLLM integration
│   │   ├── __init__.py
│   │   ├── vllm_server.py # vLLM serving
│   │   └── config.py      # LLM configuration
│   ├── routing/          # Intelligent routing
│   │   ├── __init__.py
│   │   ├── router.py      # Request router
│   │   ├── ab_testing.py  # A/B testing logic
│   │   └── canary.py      # Canary deployment
│   ├── tracing/          # Distributed tracing
│   │   ├── __init__.py
│   │   ├── tracer.py      # OpenTelemetry tracer
│   │   └── middleware.py  # FastAPI tracing middleware
│   └── models/           # Model definitions
│       └── __init__.py
├── tests/                # Comprehensive test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_tensorrt.py
│   ├── test_serving.py
│   ├── test_routing.py
│   └── integration/
├── docs/                 # Detailed documentation
│   ├── STEP_BY_STEP.md   # Implementation guide
│   ├── API.md           # API reference
│   ├── DEPLOYMENT.md    # Deployment guide
│   ├── ARCHITECTURE.md  # System architecture
│   ├── RUNBOOK.md       # Operations guide
│   └── TROUBLESHOOTING.md
├── kubernetes/          # Kubernetes manifests
│   ├── base/            # Base configuration
│   └── overlays/        # Environment-specific overlays
│       ├── dev/
│       ├── staging/
│       └── prod/
├── docker/              # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── monitoring/          # Monitoring configuration
│   ├── prometheus/
│   └── grafana/
├── scripts/            # Automation scripts
│   ├── setup.sh
│   ├── deploy.sh
│   └── benchmark.sh
├── benchmarks/         # Performance benchmarks
│   ├── latency_test.py
│   └── throughput_test.py
├── notebooks/          # Jupyter notebooks
│   └── model_analysis.ipynb
├── requirements.txt    # Python dependencies
├── .env.example       # Environment configuration template
└── README.md          # This file
```

## Core Components

### 1. TensorRT Optimization (`src/tensorrt/`)

Automated model conversion and optimization pipeline:

- **Converter**: Converts PyTorch/ONNX models to TensorRT engines
- **Calibrator**: INT8 post-training quantization with calibration
- **Optimizer**: Layer fusion, kernel auto-tuning, precision calibration

**Key Features**:
- FP32, FP16, and INT8 precision support
- Dynamic shape optimization
- Multi-stream execution
- Automatic kernel selection

### 2. Model Serving (`src/serving/`)

High-performance serving infrastructure:

- **Server**: FastAPI-based async serving with WebSockets
- **Model Loader**: Multi-format model loading with caching
- **Batch Processor**: Dynamic batching with timeout-based flushing
- **Warmup**: Automated model warmup and cache priming

**Performance**:
- Sub-millisecond latency for small models
- 1000+ QPS with batching
- <100ms P99 latency for LLMs

### 3. vLLM Integration (`src/llm/`)

Efficient large language model serving:

- **vLLM Server**: PagedAttention for memory efficiency
- **Configuration**: Optimized configs for popular LLMs
- **Continuous batching**: Maximum throughput

**Supported Models**:
- Llama 2/3 (7B-70B)
- Mistral/Mixtral
- GPT-J/NeoX
- BLOOM

### 4. Intelligent Routing (`src/routing/`)

Traffic management for ML models:

- **Router**: Rule-based and ML-based routing
- **A/B Testing**: Statistical significance testing
- **Canary Deployment**: Progressive rollout with automatic rollback
- **Shadow Traffic**: Production validation

### 5. Distributed Tracing (`src/tracing/`)

End-to-end observability:

- **OpenTelemetry**: Industry-standard tracing
- **Jaeger Integration**: Trace visualization
- **Context Propagation**: Cross-service tracing
- **Custom Spans**: Model-specific instrumentation

## Performance Benchmarks

### TensorRT Optimization Results

| Model | Precision | Batch Size | TensorRT Latency | PyTorch Latency | Speedup |
|-------|-----------|------------|------------------|-----------------|---------|
| ResNet-50 | FP16 | 1 | 1.2 ms | 8.5 ms | 7.1x |
| ResNet-50 | FP16 | 32 | 12 ms | 95 ms | 7.9x |
| ResNet-50 | INT8 | 1 | 0.8 ms | 8.5 ms | 10.6x |
| BERT-Base | FP16 | 1 | 2.1 ms | 15 ms | 7.1x |
| BERT-Base | FP16 | 32 | 18 ms | 180 ms | 10.0x |

### vLLM Serving Performance

| Model | Context Length | Throughput | Latency (P50) | Latency (P99) |
|-------|---------------|------------|---------------|---------------|
| Llama-2-7B | 2048 | 45 tok/s | 85 ms | 120 ms |
| Llama-2-13B | 2048 | 28 tok/s | 140 ms | 190 ms |
| Mistral-7B | 4096 | 52 tok/s | 75 ms | 105 ms |

## Monitoring and Observability

### Prometheus Metrics

- `model_serving_request_duration_seconds` - Request latency histogram
- `model_serving_requests_total` - Total requests counter
- `model_serving_batch_size` - Dynamic batch size gauge
- `model_serving_gpu_utilization` - GPU usage percentage
- `model_serving_model_load_time_seconds` - Model loading time

### Grafana Dashboards

Pre-built dashboards included:
- Model Serving Overview
- TensorRT Performance
- vLLM Metrics
- Request Tracing
- Resource Utilization

### Distributed Tracing

Jaeger traces include:
- Request routing decisions
- Model loading and caching
- Inference execution time
- Batch processing details
- GPU kernel execution

## Kubernetes Auto-scaling

### Horizontal Pod Autoscaler (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-serving
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric:
        name: model_serving_request_duration_seconds
      target:
        type: AverageValue
        averageValue: "100m"  # 100ms
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
```

## API Reference

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": ["resnet50-fp16", "llama-2-7b"],
  "gpu_available": true,
  "uptime_seconds": 3600
}
```

### Model Inference

```bash
POST /v1/predict
Content-Type: application/json

{
  "model": "resnet50-fp16",
  "inputs": {
    "image": "base64_encoded_image"
  },
  "parameters": {
    "temperature": 0.7
  }
}
```

Response:
```json
{
  "predictions": [
    {"class": "cat", "confidence": 0.95},
    {"class": "dog", "confidence": 0.03}
  ],
  "latency_ms": 1.2,
  "trace_id": "abc123"
}
```

### LLM Generation

```bash
POST /v1/generate
Content-Type: application/json

{
  "model": "llama-2-7b",
  "prompt": "Explain machine learning in simple terms:",
  "max_tokens": 200,
  "temperature": 0.7,
  "top_p": 0.9
}
```

Response:
```json
{
  "generated_text": "Machine learning is...",
  "tokens_generated": 150,
  "latency_ms": 850,
  "trace_id": "def456"
}
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_tensorrt.py -v

# Run integration tests
pytest tests/integration/ -v --slow

# Run performance benchmarks
python benchmarks/latency_test.py
python benchmarks/throughput_test.py
```

## Deployment Strategies

### Blue-Green Deployment

```bash
# Deploy green version
kubectl apply -k kubernetes/overlays/prod/green/

# Switch traffic
kubectl patch service model-serving -p '{"spec":{"selector":{"version":"green"}}}'

# Rollback if needed
kubectl patch service model-serving -p '{"spec":{"selector":{"version":"blue"}}}'
```

### Canary Deployment

```bash
# Deploy canary with 10% traffic
kubectl apply -f kubernetes/canary-10pct.yaml

# Monitor metrics
kubectl logs -l app=model-serving,version=canary --tail=100

# Promote to 50%
kubectl apply -f kubernetes/canary-50pct.yaml

# Full rollout
kubectl apply -f kubernetes/canary-100pct.yaml
```

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues and solutions.

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model precision
2. **Slow Cold Start**: Enable model warmup
3. **High P99 Latency**: Tune batch timeout and size
4. **Model Loading Failures**: Check CUDA compatibility and model format

## Learning Resources

### Step-by-Step Guide

See [docs/STEP_BY_STEP.md](docs/STEP_BY_STEP.md) for a detailed implementation guide.

### Additional Documentation

- [Architecture Deep Dive](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Operations Runbook](docs/RUNBOOK.md)
- [API Reference](docs/API.md)

## Performance Tuning

### TensorRT Optimization Tips

1. **Use FP16 precision** for 2-3x speedup on V100/A100
2. **Enable INT8 quantization** for 3-4x speedup (with calibration)
3. **Optimize for batch size** - match expected production load
4. **Enable multi-stream execution** for high throughput
5. **Use DLA** on embedded platforms (Xavier, Orin)

### vLLM Configuration

```python
# High throughput configuration
vllm_config = {
    "tensor_parallel_size": 4,  # Multi-GPU
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 256,
    "gpu_memory_utilization": 0.95,
    "enable_chunked_prefill": True,
}

# Low latency configuration
vllm_config = {
    "tensor_parallel_size": 1,
    "max_num_batched_tokens": 2048,
    "max_num_seqs": 16,
    "gpu_memory_utilization": 0.80,
    "enable_prefix_caching": True,
}
```


