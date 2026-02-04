# API Reference - High-Performance Model Serving

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Base URL](#base-url)
- [Common Headers](#common-headers)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Metrics](#metrics)
  - [Model Inference](#model-inference)
  - [Text Generation](#text-generation)
  - [Model Management](#model-management)
- [WebSocket Streaming](#websocket-streaming)
- [Request Examples](#request-examples)
- [Response Schemas](#response-schemas)
- [Error Codes](#error-codes)

---

## Overview

The Model Serving API provides high-performance inference endpoints for machine learning models with support for:

- Multiple model formats (TensorRT, PyTorch, ONNX)
- Batch and single inference
- Streaming text generation
- Model lifecycle management
- Real-time metrics and health monitoring

**API Version**: v1
**Protocol**: HTTP/1.1, WebSocket
**Content-Type**: application/json

---

## Authentication

### API Key Authentication

All endpoints (except `/health` and `/metrics`) require authentication using API keys passed in the `Authorization` header.

```http
Authorization: Bearer YOUR_API_KEY
```

**Example**:
```bash
curl -H "Authorization: Bearer sk-abc123..." \
     http://api.example.com/v1/predict
```

### Obtaining API Keys

Contact your system administrator or use the API key management portal to generate keys.

**Security Notes**:
- Keep API keys secure and never commit them to version control
- Rotate keys regularly (recommended: every 90 days)
- Use different keys for development, staging, and production

---

## Base URL

### Production
```
https://api.model-serving.example.com
```

### Staging
```
https://staging-api.model-serving.example.com
```

### Local Development
```
http://localhost:8000
```

---

## Common Headers

### Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes* | Bearer token for authentication |
| `Content-Type` | Yes | Must be `application/json` |
| `X-Request-ID` | No | Client-provided request ID for tracking |
| `X-Model-Version` | No | Specific model version to use |

*Not required for `/health` and `/metrics` endpoints

### Response Headers

| Header | Description |
|--------|-------------|
| `X-Trace-ID` | Distributed tracing ID for request correlation |
| `X-Request-ID` | Echo of client-provided request ID |
| `X-RateLimit-Limit` | Maximum requests per time window |
| `X-RateLimit-Remaining` | Remaining requests in current window |
| `X-RateLimit-Reset` | Unix timestamp when rate limit resets |

---

## Error Handling

All errors return a consistent JSON structure:

```json
{
  "detail": "Error message describing what went wrong",
  "trace_id": "abc123...",
  "timestamp": "2024-01-15T10:30:00Z",
  "path": "/v1/predict",
  "request_id": "req_12345"
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request succeeded |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Missing or invalid API key |
| 404 | Not Found | Model or endpoint not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Service temporarily unavailable |

---

## Rate Limiting

Rate limits are enforced per API key:

| Endpoint | Rate Limit |
|----------|------------|
| `/v1/predict` | 100 requests/minute |
| `/v1/generate` | 50 requests/minute |
| `/v1/models/*` | 20 requests/minute |

When rate limited, the API returns:

```json
{
  "detail": "Rate limit exceeded. Try again in 30 seconds.",
  "retry_after": 30
}
```

**Rate Limit Headers**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1705318800
```

---

## Endpoints

### Health Check

Check service health and readiness.

**Endpoint**: `GET /health`

**Authentication**: Not required

**Response**: 200 OK
```json
{
  "status": "healthy",
  "models_loaded": [
    "resnet50-fp16",
    "llama-2-7b"
  ],
  "gpu_available": true,
  "uptime_seconds": 3600.5,
  "version": "1.0.0"
}
```

**Response Fields**:
- `status` (string): Health status ("healthy" or "unhealthy")
- `models_loaded` (array): List of currently loaded model names
- `gpu_available` (boolean): Whether GPU is accessible
- `uptime_seconds` (float): Service uptime in seconds
- `version` (string): API version

**Example**:
```bash
curl http://localhost:8000/health
```

---

### Metrics

Get Prometheus-formatted metrics.

**Endpoint**: `GET /metrics`

**Authentication**: Not required

**Response**: 200 OK (text/plain)
```
# HELP model_serving_requests_total Total number of requests
# TYPE model_serving_requests_total counter
model_serving_requests_total{model="resnet50-fp16",endpoint="/v1/predict",status="success"} 1234

# HELP model_serving_request_duration_seconds Request duration in seconds
# TYPE model_serving_request_duration_seconds histogram
model_serving_request_duration_seconds_bucket{model="resnet50-fp16",endpoint="/v1/predict",le="0.001"} 45
model_serving_request_duration_seconds_bucket{model="resnet50-fp16",endpoint="/v1/predict",le="0.005"} 123
...
```

**Available Metrics**:

| Metric | Type | Description |
|--------|------|-------------|
| `model_serving_requests_total` | Counter | Total requests by model, endpoint, status |
| `model_serving_request_duration_seconds` | Histogram | Request latency distribution |
| `model_serving_batch_size` | Histogram | Batch size distribution |
| `model_serving_active_requests` | Gauge | Currently active requests |
| `model_serving_model_load_time_seconds` | Histogram | Model loading time |

**Example**:
```bash
curl http://localhost:8000/metrics
```

---

### Model Inference

Perform inference on a loaded model.

**Endpoint**: `POST /v1/predict`

**Authentication**: Required

**Request Body**:
```json
{
  "model": "string",
  "inputs": {
    "key": "value"
  },
  "parameters": {
    "key": "value"
  }
}
```

**Request Fields**:
- `model` (string, required): Model name or identifier
- `inputs` (object, required): Model inputs (format varies by model)
- `parameters` (object, optional): Inference parameters

**Response**: 200 OK
```json
{
  "predictions": [],
  "latency_ms": 1.2,
  "model": "resnet50-fp16",
  "trace_id": "abc123..."
}
```

**Response Fields**:
- `predictions` (array|object): Model predictions
- `latency_ms` (float): Inference latency in milliseconds
- `model` (string): Model that processed the request
- `trace_id` (string): Trace ID for debugging

**Examples**:

#### Image Classification
```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "resnet50-fp16",
    "inputs": {
      "image": "base64_encoded_image_data_here..."
    },
    "parameters": {
      "top_k": 5
    }
  }'
```

**Response**:
```json
{
  "predictions": [
    {"class": "golden_retriever", "confidence": 0.945},
    {"class": "labrador", "confidence": 0.032},
    {"class": "dog", "confidence": 0.015},
    {"class": "puppy", "confidence": 0.005},
    {"class": "animal", "confidence": 0.003}
  ],
  "latency_ms": 1.2,
  "model": "resnet50-fp16",
  "trace_id": "trace_1705318800123"
}
```

#### Object Detection
```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "yolov8-fp16",
    "inputs": {
      "image": "base64_encoded_image..."
    },
    "parameters": {
      "confidence_threshold": 0.5,
      "iou_threshold": 0.45
    }
  }'
```

**Response**:
```json
{
  "predictions": {
    "boxes": [
      {
        "bbox": [100, 150, 200, 300],
        "class": "person",
        "confidence": 0.95
      },
      {
        "bbox": [350, 200, 450, 400],
        "class": "dog",
        "confidence": 0.89
      }
    ],
    "num_detections": 2
  },
  "latency_ms": 3.5,
  "model": "yolov8-fp16",
  "trace_id": "trace_1705318800456"
}
```

#### Text Embedding
```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "bert-base",
    "inputs": {
      "text": "This is a sample sentence for embedding."
    }
  }'
```

**Response**:
```json
{
  "predictions": {
    "embeddings": [0.123, -0.456, 0.789, ...],
    "dimension": 768
  },
  "latency_ms": 2.1,
  "model": "bert-base",
  "trace_id": "trace_1705318800789"
}
```

**Error Responses**:

**400 Bad Request** - Invalid input:
```json
{
  "detail": "Invalid input format: 'image' field must be base64 encoded",
  "trace_id": "trace_xxx"
}
```

**404 Not Found** - Model not found:
```json
{
  "detail": "Model 'invalid-model' not found or not loaded",
  "trace_id": "trace_xxx"
}
```

**500 Internal Server Error** - Inference failed:
```json
{
  "detail": "Prediction failed: CUDA out of memory",
  "trace_id": "trace_xxx"
}
```

---

### Text Generation

Generate text completions using Large Language Models.

**Endpoint**: `POST /v1/generate`

**Authentication**: Required

**Request Body**:
```json
{
  "model": "string",
  "prompt": "string",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "stop_sequences": ["string"]
}
```

**Request Fields**:
- `model` (string, required): LLM model name
- `prompt` (string, required): Input prompt text
- `max_tokens` (integer, optional): Maximum tokens to generate (default: 100, max: 2048)
- `temperature` (float, optional): Sampling temperature (default: 0.7, range: 0.0-2.0)
- `top_p` (float, optional): Nucleus sampling parameter (default: 0.9, range: 0.0-1.0)
- `top_k` (integer, optional): Top-k sampling parameter (default: 50, range: 1-100)
- `stop_sequences` (array, optional): Sequences to stop generation

**Response**: 200 OK
```json
{
  "generated_text": "string",
  "tokens_generated": 150,
  "latency_ms": 850.5,
  "model": "llama-2-7b",
  "trace_id": "abc123..."
}
```

**Response Fields**:
- `generated_text` (string): Generated text completion
- `tokens_generated` (integer): Number of tokens generated
- `latency_ms` (float): Generation latency in milliseconds
- `model` (string): Model used for generation
- `trace_id` (string): Trace ID for debugging

**Examples**:

#### Simple Completion
```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "llama-2-7b",
    "prompt": "Explain quantum computing in simple terms:",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

**Response**:
```json
{
  "generated_text": "Quantum computing is a revolutionary technology that uses the principles of quantum mechanics to process information. Unlike classical computers that use bits (0s and 1s), quantum computers use quantum bits or 'qubits' that can exist in multiple states simultaneously through a phenomenon called superposition. This allows quantum computers to perform certain calculations exponentially faster than classical computers, particularly for problems involving optimization, cryptography, and molecular simulation.",
  "tokens_generated": 87,
  "latency_ms": 1245.3,
  "model": "llama-2-7b",
  "trace_id": "trace_1705318801234"
}
```

#### With Stop Sequences
```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "llama-2-7b",
    "prompt": "Q: What is the capital of France?\nA:",
    "max_tokens": 50,
    "temperature": 0.3,
    "stop_sequences": ["\n", "Q:"]
  }'
```

**Response**:
```json
{
  "generated_text": " The capital of France is Paris.",
  "tokens_generated": 8,
  "latency_ms": 125.7,
  "model": "llama-2-7b",
  "trace_id": "trace_1705318801567"
}
```

#### Creative Writing
```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "llama-2-7b",
    "prompt": "Write a haiku about artificial intelligence:",
    "max_tokens": 100,
    "temperature": 0.9,
    "top_p": 0.95
  }'
```

**Response**:
```json
{
  "generated_text": "Silicon minds awake\nThinking without consciousness\nFuture unfolds now",
  "tokens_generated": 15,
  "latency_ms": 234.1,
  "model": "llama-2-7b",
  "trace_id": "trace_1705318801890"
}
```

**Error Responses**:

**400 Bad Request** - Invalid parameters:
```json
{
  "detail": "max_tokens must be between 1 and 2048",
  "trace_id": "trace_xxx"
}
```

**404 Not Found** - Model not available:
```json
{
  "detail": "Model 'llama-2-7b' not found or not loaded",
  "trace_id": "trace_xxx"
}
```

---

### Model Management

#### Load Model

Load a model into memory for serving.

**Endpoint**: `POST /v1/models/{model_name}/load`

**Authentication**: Required

**Path Parameters**:
- `model_name` (string): Model identifier

**Query Parameters**:
- `model_format` (string, optional): Model format ("tensorrt", "pytorch", "onnx"), default: "tensorrt"

**Response**: 200 OK
```json
{
  "model": "resnet50-fp16",
  "format": "tensorrt",
  "status": "loaded",
  "load_time_seconds": 2.3
}
```

**Example**:
```bash
curl -X POST "http://localhost:8000/v1/models/resnet50-fp16/load?model_format=tensorrt" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response**:
```json
{
  "model": "resnet50-fp16",
  "format": "tensorrt",
  "status": "loaded",
  "load_time_seconds": 2.34
}
```

**Error Responses**:

**404 Not Found**:
```json
{
  "detail": "Model file not found: /models/resnet50-fp16.trt",
  "trace_id": "trace_xxx"
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Failed to load model: CUDA initialization failed",
  "trace_id": "trace_xxx"
}
```

---

#### Unload Model

Unload a model from memory.

**Endpoint**: `POST /v1/models/{model_name}/unload`

**Authentication**: Required

**Path Parameters**:
- `model_name` (string): Model identifier

**Response**: 200 OK
```json
{
  "model": "resnet50-fp16",
  "status": "unloaded"
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/v1/models/resnet50-fp16/unload \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Error Responses**:

**404 Not Found**:
```json
{
  "detail": "Model 'resnet50-fp16' not found or not loaded",
  "trace_id": "trace_xxx"
}
```

---

#### List Models

List all currently loaded models.

**Endpoint**: `GET /v1/models`

**Authentication**: Required

**Response**: 200 OK
```json
{
  "models": [
    {
      "name": "resnet50-fp16",
      "format": "tensorrt",
      "loaded_at": "2024-01-15T10:30:00Z",
      "memory_usage_mb": 256.5
    },
    {
      "name": "llama-2-7b",
      "format": "vllm",
      "loaded_at": "2024-01-15T10:31:00Z",
      "memory_usage_mb": 14336.0
    }
  ],
  "total_models": 2
}
```

**Example**:
```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## WebSocket Streaming

For streaming text generation, use WebSocket connections.

**Endpoint**: `ws://localhost:8000/v1/generate/stream`

**Authentication**: Pass API key as query parameter: `?api_key=YOUR_API_KEY`

**Client Example** (JavaScript):
```javascript
const ws = new WebSocket('ws://localhost:8000/v1/generate/stream?api_key=YOUR_API_KEY');

ws.onopen = () => {
  ws.send(JSON.stringify({
    model: 'llama-2-7b',
    prompt: 'Write a story about AI:',
    max_tokens: 500,
    temperature: 0.8
  }));
};

ws.onmessage = (event) => {
  const chunk = JSON.parse(event.data);
  if (chunk.type === 'token') {
    process.stdout.write(chunk.text);
  } else if (chunk.type === 'done') {
    console.log(`\nGenerated ${chunk.tokens_generated} tokens in ${chunk.latency_ms}ms`);
    ws.close();
  } else if (chunk.type === 'error') {
    console.error('Error:', chunk.message);
    ws.close();
  }
};
```

**Python Example**:
```python
import asyncio
import websockets
import json

async def generate_stream():
    uri = "ws://localhost:8000/v1/generate/stream?api_key=YOUR_API_KEY"

    async with websockets.connect(uri) as websocket:
        # Send request
        await websocket.send(json.dumps({
            "model": "llama-2-7b",
            "prompt": "Write a poem about technology:",
            "max_tokens": 200,
            "temperature": 0.9
        }))

        # Receive tokens
        while True:
            message = await websocket.recv()
            chunk = json.loads(message)

            if chunk["type"] == "token":
                print(chunk["text"], end="", flush=True)
            elif chunk["type"] == "done":
                print(f"\n\nCompleted: {chunk['tokens_generated']} tokens")
                break
            elif chunk["type"] == "error":
                print(f"Error: {chunk['message']}")
                break

asyncio.run(generate_stream())
```

**Message Types**:

**Token** (streaming):
```json
{
  "type": "token",
  "text": " the",
  "index": 5
}
```

**Done**:
```json
{
  "type": "done",
  "tokens_generated": 150,
  "latency_ms": 2345.6,
  "trace_id": "trace_xxx"
}
```

**Error**:
```json
{
  "type": "error",
  "message": "Generation failed: model timeout",
  "trace_id": "trace_xxx"
}
```

---

## Request Examples

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Image classification
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d @request.json

# Text generation
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "llama-2-7b",
    "prompt": "Explain recursion:",
    "max_tokens": 150
  }'

# Load model
curl -X POST "http://localhost:8000/v1/models/bert-base/load?model_format=pytorch" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Python Requests

```python
import requests
import base64

API_KEY = "YOUR_API_KEY"
BASE_URL = "http://localhost:8000"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Image classification
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post(
    f"{BASE_URL}/v1/predict",
    headers=headers,
    json={
        "model": "resnet50-fp16",
        "inputs": {"image": image_data},
        "parameters": {"top_k": 5}
    }
)

print(response.json())

# Text generation
response = requests.post(
    f"{BASE_URL}/v1/generate",
    headers=headers,
    json={
        "model": "llama-2-7b",
        "prompt": "What is machine learning?",
        "max_tokens": 200,
        "temperature": 0.7
    }
)

print(response.json())
```

### JavaScript Fetch

```javascript
const API_KEY = 'YOUR_API_KEY';
const BASE_URL = 'http://localhost:8000';

// Health check
fetch(`${BASE_URL}/health`)
  .then(res => res.json())
  .then(data => console.log(data));

// Inference
fetch(`${BASE_URL}/v1/predict`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    model: 'resnet50-fp16',
    inputs: { image: imageBase64 },
    parameters: { top_k: 5 }
  })
})
  .then(res => res.json())
  .then(data => console.log(data));
```

---

## Response Schemas

### PredictResponse

```typescript
{
  predictions: Array<any> | Object | number[],
  latency_ms: number,
  model: string,
  trace_id?: string
}
```

### GenerateResponse

```typescript
{
  generated_text: string,
  tokens_generated: number,
  latency_ms: number,
  model: string,
  trace_id?: string
}
```

### HealthResponse

```typescript
{
  status: "healthy" | "unhealthy",
  models_loaded: string[],
  gpu_available: boolean,
  uptime_seconds: number,
  version: string
}
```

### ErrorResponse

```typescript
{
  detail: string,
  trace_id?: string,
  timestamp?: string,
  path?: string,
  request_id?: string
}
```

---

## Error Codes

| Error Code | HTTP Status | Description | Solution |
|------------|-------------|-------------|----------|
| `INVALID_API_KEY` | 401 | Invalid or missing API key | Check Authorization header |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests | Wait and retry |
| `MODEL_NOT_FOUND` | 404 | Model not found | Check model name or load model |
| `INVALID_INPUT` | 400 | Invalid request body | Validate input format |
| `INFERENCE_FAILED` | 500 | Model inference error | Check logs, retry |
| `OUT_OF_MEMORY` | 500 | GPU/CPU out of memory | Reduce batch size |
| `MODEL_LOAD_FAILED` | 500 | Failed to load model | Check model file and format |
| `TIMEOUT` | 504 | Request timeout | Reduce input size or retry |

---

## Additional Resources

- [Step-by-Step Implementation Guide](STEP_BY_STEP.md)
- [Architecture Documentation](ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Operations Runbook](RUNBOOK.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)

---
