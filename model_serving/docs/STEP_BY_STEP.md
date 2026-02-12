# Project 202: High-Performance Model Serving - Step-by-Step Implementation Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Step 1: Environment Setup](#step-1-environment-setup)
4. [Step 2: TensorRT Model Conversion](#step-2-tensorrt-model-conversion)
5. [Step 3: Implementing the Serving Layer](#step-3-implementing-the-serving-layer)
6. [Step 4: Adding vLLM Support](#step-4-adding-vllm-support)
7. [Step 5: Setting Up Routing](#step-5-setting-up-routing)
8. [Step 6: Distributed Tracing Integration](#step-6-distributed-tracing-integration)
9. [Step 7: Kubernetes Deployment](#step-7-kubernetes-deployment)
10. [Step 8: Monitoring and Observability](#step-8-monitoring-and-observability)
11. [Step 9: Production Hardening](#step-9-production-hardening)
12. [Common Pitfalls and Debugging](#common-pitfalls-and-debugging)
13. [Testing and Validation](#testing-and-validation)
14. [Next Steps](#next-steps)

---

## Introduction

This guide walks you through building a production-ready, high-performance model serving system from scratch. By the end, you'll have:

- TensorRT-optimized model inference (5-10x faster than PyTorch)
- vLLM integration for efficient LLM serving
- Intelligent request routing with health checking
- Distributed tracing for observability
- Auto-scaling Kubernetes deployment
- Comprehensive monitoring and alerting

**Estimated Time**: 6-8 hours for complete implementation

**Difficulty Level**: Advanced (Senior Engineer)

---

## Prerequisites

### Required Knowledge

- Strong Python programming skills
- Experience with FastAPI or similar web frameworks
- Understanding of ML model inference
- Basic Kubernetes knowledge
- Familiarity with Docker and containerization

### System Requirements

```bash
# Hardware
- NVIDIA GPU (V100, A100, or newer recommended)
- CUDA 12.1 or later
- 16GB+ system RAM
- 50GB+ free disk space

# Software
- Ubuntu 20.04+ or similar Linux distribution
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (for production deployment)
- Python 3.10+
```

### Account Requirements

- Docker Hub account (or private registry)
- Kubernetes cluster (AWS EKS, GCP GKE, or local K3s)
- (Optional) Hugging Face account for model downloads

---

## Step 1: Environment Setup

### 1.1 Install System Dependencies

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install CUDA (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Verify CUDA installation
nvidia-smi
nvcc --version

# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access from Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu20.04 nvidia-smi
```

**Validation**: You should see your GPU listed in the `nvidia-smi` output.

### 1.2 Create Project Structure

```bash
# Create project directory
mkdir -p ~/model-serving-project && cd ~/model-serving-project

# Create directory structure
mkdir -p {src/{tensorrt,serving,llm,routing,tracing,models},tests,docs,kubernetes,docker,scripts,models,monitoring}

# Initialize Python virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 1.3 Install Python Dependencies

Create `requirements.txt`:

```txt
# Core dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
httpx==0.25.1

# ML and optimization
torch==2.1.0
tensorrt==8.6.1
onnx==1.15.0
onnxruntime==1.16.3
vllm==0.2.7

# Monitoring and tracing
prometheus-client==0.19.0
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0
opentelemetry-instrumentation-fastapi==0.42b0
opentelemetry-exporter-jaeger==1.21.0

# Utilities
numpy==1.24.3
pillow==10.1.0
python-dotenv==1.0.0
pyyaml==6.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
httpx==0.25.1  # For testing HTTP clients

# Development tools
black==23.11.0
ruff==0.1.6
mypy==1.7.1
```

Install dependencies:

```bash
pip install -r requirements.txt

# Install TensorRT Python bindings (if not included)
pip install nvidia-tensorrt
```

### 1.4 Configure Environment Variables

Create `.env` file:

```bash
# Model serving configuration
MODEL_CACHE_DIR=/tmp/model_cache
MAX_BATCH_SIZE=32
BATCH_TIMEOUT_MS=10
DEFAULT_MODEL=resnet50-fp16

# Server configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_WORKERS=1
LOG_LEVEL=INFO

# GPU configuration
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_FRACTION=0.9

# TensorRT configuration
TRT_OPTIMIZATION_LEVEL=5
TRT_PRECISION=fp16
TRT_TIMING_CACHE=/tmp/timing_cache.bin

# vLLM configuration
VLLM_MODEL=meta-llama/Llama-2-7b-hf
VLLM_TENSOR_PARALLEL=1
VLLM_MAX_NUM_SEQS=256
VLLM_GPU_MEMORY_UTILIZATION=0.95

# Tracing configuration
JAEGER_ENABLED=true
JAEGER_HOST=localhost
JAEGER_PORT=6831
TRACE_SAMPLE_RATE=0.1

# Monitoring
PROMETHEUS_PORT=9090
METRICS_ENABLED=true
```

**Validation**: Run `source .env` to load environment variables.

---

## Step 2: TensorRT Model Conversion

TensorRT provides 5-10x inference speedup through layer fusion, precision calibration, and kernel auto-tuning.

### 2.1 Create the Converter Module

Create `src/tensorrt/__init__.py`:

```python
"""TensorRT optimization module."""

from .converter import TensorRTConverter, PrecisionMode, ConversionConfig
from .calibrator import INT8Calibrator
from .optimizer import TensorRTOptimizer

__all__ = [
    'TensorRTConverter',
    'PrecisionMode',
    'ConversionConfig',
    'INT8Calibrator',
    'TensorRTOptimizer',
]
```

Create `src/tensorrt/converter.py` (see full implementation in source).

### 2.2 Implement INT8 Calibration

For INT8 quantization, we need a calibration dataset. Create `src/tensorrt/calibrator.py`:

```python
"""
INT8 Calibration for TensorRT

Implements post-training quantization calibration for INT8 precision mode.
"""

import logging
import numpy as np
import tensorrt as trt
from typing import List, Optional, Dict, Any
import pycuda.driver as cuda
import pycuda.autoinit

logger = logging.getLogger(__name__)


class INT8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 calibration using entropy calibration v2.

    This calibrator processes a calibration dataset to determine optimal
    quantization ranges for INT8 inference.
    """

    def __init__(
        self,
        calibration_data: List[np.ndarray],
        cache_file: str = "/tmp/int8_calibration.cache",
        batch_size: int = 1,
    ):
        """
        Initialize INT8 calibrator.

        Args:
            calibration_data: List of calibration input samples
            cache_file: Path to save/load calibration cache
            batch_size: Calibration batch size
        """
        super().__init__()

        self.calibration_data = calibration_data
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate device memory for calibration batch
        self.device_input = None

        logger.info(
            f"INT8Calibrator initialized with {len(calibration_data)} samples, "
            f"batch_size={batch_size}"
        )

    def get_batch_size(self) -> int:
        """Return batch size for calibration."""
        return self.batch_size

    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        """
        Get next calibration batch.

        Args:
            names: List of input tensor names

        Returns:
            List of device memory pointers or None if done
        """
        if self.current_index >= len(self.calibration_data):
            logger.info("Calibration complete")
            return None

        # Get batch
        batch = self.calibration_data[self.current_index]
        self.current_index += 1

        # Allocate device memory on first batch
        if self.device_input is None:
            self.device_input = cuda.mem_alloc(batch.nbytes)

        # Copy batch to device
        cuda.memcpy_htod(self.device_input, batch)

        logger.debug(f"Calibration batch {self.current_index}/{len(self.calibration_data)}")

        return [int(self.device_input)]

    def read_calibration_cache(self) -> bytes:
        """Read calibration cache from disk."""
        try:
            with open(self.cache_file, "rb") as f:
                cache = f.read()
                logger.info(f"Loaded calibration cache from {self.cache_file}")
                return cache
        except FileNotFoundError:
            logger.info("No calibration cache found, will calibrate from scratch")
            return b""

    def write_calibration_cache(self, cache: bytes) -> None:
        """Write calibration cache to disk."""
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        logger.info(f"Saved calibration cache to {self.cache_file}")


def create_calibration_dataset(
    num_samples: int = 100,
    input_shape: tuple = (1, 3, 224, 224),
) -> List[np.ndarray]:
    """
    Create a synthetic calibration dataset.

    In production, use real data samples from your training/validation set.

    Args:
        num_samples: Number of calibration samples
        input_shape: Input tensor shape

    Returns:
        List of calibration samples
    """
    logger.info(f"Creating calibration dataset with {num_samples} samples")

    calibration_data = []
    for i in range(num_samples):
        # Generate random data (use real data in production)
        sample = np.random.randn(*input_shape).astype(np.float32)
        calibration_data.append(sample)

    return calibration_data
```

### 2.3 Convert a Model to TensorRT

Create a conversion script `scripts/convert_model.py`:

```python
#!/usr/bin/env python3
"""
Model conversion script for TensorRT optimization.

Usage:
    python scripts/convert_model.py \
        --model resnet50 \
        --precision fp16 \
        --output models/resnet50-fp16.trt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torchvision.models as models

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorrt.converter import TensorRTConverter, PrecisionMode, ConversionConfig
from tensorrt.calibrator import INT8Calibrator, create_calibration_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_name: str) -> torch.nn.Module:
    """Load a PyTorch model by name."""
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TensorRT")
    parser.add_argument("--model", required=True, help="Model name (resnet50, resnet18, etc.)")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Precision mode"
    )
    parser.add_argument("--output", required=True, help="Output TensorRT engine path")
    parser.add_argument("--batch-size", type=int, default=1, help="Maximum batch size")
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Number of calibration samples for INT8"
    )

    args = parser.parse_args()

    logger.info(f"Converting {args.model} to TensorRT ({args.precision})")

    # Load PyTorch model
    logger.info("Loading PyTorch model...")
    model = load_model(args.model)

    # Define input shapes
    input_shapes = {"input": (args.batch_size, 3, 224, 224)}

    # Create calibrator for INT8
    calibrator = None
    if args.precision == "int8":
        logger.info("Creating INT8 calibration dataset...")
        calib_data = create_calibration_dataset(
            num_samples=args.calibration_samples,
            input_shape=(1, 3, 224, 224)
        )
        calibrator = INT8Calibrator(calib_data, batch_size=1)

    # Create conversion config
    config = ConversionConfig(
        precision=PrecisionMode(args.precision),
        max_batch_size=args.batch_size,
        max_workspace_size=1 << 30,  # 1GB
        enable_dynamic_shapes=True,
        enable_timing_cache=True,
        calibrator=calibrator,
        optimization_level=5,
    )

    # Convert model
    logger.info("Converting to TensorRT...")
    converter = TensorRTConverter(config)
    engine = converter.convert_pytorch_model(
        model=model,
        input_shapes=input_shapes,
        input_names=["input"],
        output_names=["output"],
    )

    # Save engine
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter.save_engine(engine, output_path)

    logger.info(f"TensorRT engine saved to {output_path}")

    # Validate engine
    logger.info("Validating engine...")
    test_input = torch.randn(1, 3, 224, 224).numpy()
    if converter.validate_engine(engine, {"input": test_input}):
        logger.info("✓ Engine validation passed")
    else:
        logger.error("✗ Engine validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

Make it executable and run:

```bash
chmod +x scripts/convert_model.py

# Convert ResNet-50 to FP16
python scripts/convert_model.py \
    --model resnet50 \
    --precision fp16 \
    --batch-size 32 \
    --output models/resnet50-fp16.trt

# Convert with INT8 quantization
python scripts/convert_model.py \
    --model resnet50 \
    --precision int8 \
    --batch-size 32 \
    --calibration-samples 500 \
    --output models/resnet50-int8.trt
```

**Expected Output**:
```
INFO:__main__:Converting resnet50 to TensorRT (fp16)
INFO:__main__:Loading PyTorch model...
INFO:__main__:Converting to TensorRT...
INFO:converter:Exporting PyTorch model to ONNX
INFO:converter:Parsing ONNX model
INFO:converter:Successfully parsed ONNX model with 53 layers
INFO:converter:Building TensorRT engine (this may take several minutes)...
INFO:converter:Enabled FP16 precision mode
INFO:converter:Successfully built TensorRT engine
INFO:__main__:TensorRT engine saved to models/resnet50-fp16.trt
INFO:__main__:✓ Engine validation passed
```

**Common Issues**:

- **CUDA Out of Memory**: Reduce `max_workspace_size` or batch size
- **Unsupported ONNX ops**: Update TensorRT or simplify model architecture
- **Long build times**: Normal for first build; subsequent builds use timing cache

---

## Step 3: Implementing the Serving Layer

Now we'll build the FastAPI-based serving infrastructure.

### 3.1 Create Model Loader

Create `src/serving/model_loader.py`:

```python
"""
Model Loader Module

Handles loading, caching, and lifecycle management of ML models in multiple formats.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any
import threading

import tensorrt as trt
import torch
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Multi-format model loader with caching and lifecycle management.

    Supports:
    - TensorRT engines (.trt)
    - PyTorch models (.pt, .pth)
    - ONNX models (.onnx)

    Features:
    - Model caching in memory
    - Thread-safe loading
    - Automatic resource cleanup
    - Model warmup
    """

    def __init__(self, cache_dir: str = "/tmp/model_cache"):
        """
        Initialize model loader.

        Args:
            cache_dir: Directory for cached models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._models: Dict[str, Any] = {}
        self._contexts: Dict[str, trt.IExecutionContext] = {}
        self._lock = threading.RLock()

        logger.info(f"ModelLoader initialized with cache_dir={cache_dir}")

    def load_model(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        model_format: str = "tensorrt",
    ) -> Any:
        """
        Load a model into memory.

        Args:
            model_name: Unique model identifier
            model_path: Path to model file (optional if in cache)
            model_format: Model format (tensorrt, pytorch, onnx)

        Returns:
            Loaded model object

        Raises:
            FileNotFoundError: If model file not found
            ValueError: If unsupported format
        """
        with self._lock:
            # Check if already loaded
            if model_name in self._models:
                logger.info(f"Model {model_name} already loaded")
                return self._models[model_name]

            # Determine model path
            if model_path is None:
                model_path = self._find_model_in_cache(model_name, model_format)

            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            logger.info(f"Loading model {model_name} from {model_path} ({model_format})")

            # Load based on format
            if model_format == "tensorrt":
                model = self._load_tensorrt(model_name, model_path)
            elif model_format == "pytorch":
                model = self._load_pytorch(model_path)
            elif model_format == "onnx":
                model = self._load_onnx(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_format}")

            self._models[model_name] = model
            logger.info(f"Model {model_name} loaded successfully")

            return model

    def _load_tensorrt(self, model_name: str, model_path: str) -> trt.ICudaEngine:
        """Load TensorRT engine."""
        trt_logger = trt.Logger(trt.Logger.WARNING)

        with open(model_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(engine_data)

        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {model_path}")

        # Create execution context
        context = engine.create_execution_context()
        self._contexts[model_name] = context

        logger.info(f"TensorRT engine loaded: {engine.num_bindings} bindings")

        return engine

    def _load_pytorch(self, model_path: str) -> torch.nn.Module:
        """Load PyTorch model."""
        model = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        return model

    def _load_onnx(self, model_path: str) -> Any:
        """Load ONNX model."""
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)

        return session

    def _find_model_in_cache(self, model_name: str, model_format: str) -> str:
        """Find model file in cache directory."""
        extensions = {
            "tensorrt": [".trt", ".engine"],
            "pytorch": [".pt", ".pth"],
            "onnx": [".onnx"],
        }

        for ext in extensions.get(model_format, []):
            model_path = self.cache_dir / f"{model_name}{ext}"
            if model_path.exists():
                return str(model_path)

        raise FileNotFoundError(f"Model {model_name} not found in cache")

    def get_model(self, model_name: str) -> Any:
        """
        Get a loaded model.

        Args:
            model_name: Model identifier

        Returns:
            Model object

        Raises:
            KeyError: If model not loaded
        """
        if model_name not in self._models:
            raise KeyError(f"Model {model_name} not loaded")
        return self._models[model_name]

    def get_context(self, model_name: str) -> Optional[trt.IExecutionContext]:
        """Get TensorRT execution context for a model."""
        return self._contexts.get(model_name)

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self._models

    def list_loaded_models(self) -> list:
        """List all loaded model names."""
        return list(self._models.keys())

    def unload_model(self, model_name: str) -> None:
        """
        Unload a model from memory.

        Args:
            model_name: Model to unload
        """
        with self._lock:
            if model_name in self._models:
                del self._models[model_name]
                logger.info(f"Model {model_name} unloaded")

            if model_name in self._contexts:
                del self._contexts[model_name]

    def unload_all_models(self) -> None:
        """Unload all models."""
        with self._lock:
            model_names = list(self._models.keys())
            for name in model_names:
                self.unload_model(name)
            logger.info("All models unloaded")
```

### 3.2 Implement Dynamic Batching

Create `src/serving/batch_processor.py`:

```python
"""
Dynamic Batch Processor

Implements dynamic batching to optimize GPU utilization by grouping
inference requests together.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Represents a single request in the batch queue."""
    request_id: str
    inputs: Any
    future: asyncio.Future
    timestamp: float


class DynamicBatchProcessor:
    """
    Dynamic batching processor for inference optimization.

    Collects incoming requests and batches them together to maximize
    GPU utilization. Uses timeout-based flushing to balance latency
    and throughput.

    Attributes:
        max_batch_size: Maximum requests per batch
        timeout_ms: Maximum wait time before flushing batch
        max_queue_size: Maximum queued requests
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        timeout_ms: float = 10.0,
        max_queue_size: int = 1000,
    ):
        """
        Initialize batch processor.

        Args:
            max_batch_size: Maximum batch size
            timeout_ms: Timeout for batch flushing (milliseconds)
            max_queue_size: Maximum queue size
        """
        self.max_batch_size = max_batch_size
        self.timeout_s = timeout_ms / 1000.0
        self.max_queue_size = max_queue_size

        self._queue: Deque[BatchRequest] = deque()
        self._lock = asyncio.Lock()
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"DynamicBatchProcessor initialized: "
            f"max_batch_size={max_batch_size}, timeout_ms={timeout_ms}"
        )

    async def start(self) -> None:
        """Start the batch processing loop."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_loop())
        logger.info("Batch processor started")

    async def shutdown(self) -> None:
        """Shutdown the batch processor."""
        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        logger.info("Batch processor shutdown complete")

    async def add_request(
        self,
        request_id: str,
        inputs: Any,
    ) -> Any:
        """
        Add a request to the batch queue.

        Args:
            request_id: Unique request identifier
            inputs: Request input data

        Returns:
            Inference result (awaitable)

        Raises:
            RuntimeError: If queue is full
        """
        async with self._lock:
            if len(self._queue) >= self.max_queue_size:
                raise RuntimeError("Batch queue is full")

            future = asyncio.Future()
            request = BatchRequest(
                request_id=request_id,
                inputs=inputs,
                future=future,
                timestamp=time.time(),
            )

            self._queue.append(request)

        # Wait for result
        return await future

    async def _process_loop(self) -> None:
        """Background task for processing batches."""
        logger.info("Batch processing loop started")

        while self._running:
            try:
                await asyncio.sleep(self.timeout_s)
                await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing error: {e}", exc_info=True)

        logger.info("Batch processing loop stopped")

    async def _flush_batch(self) -> None:
        """Flush current batch and process requests."""
        async with self._lock:
            if not self._queue:
                return

            # Collect batch
            batch_size = min(len(self._queue), self.max_batch_size)
            batch = [self._queue.popleft() for _ in range(batch_size)]

        if not batch:
            return

        logger.debug(f"Processing batch of size {len(batch)}")

        try:
            # Process batch
            results = await self._process_batch(batch)

            # Set results
            for request, result in zip(batch, results):
                if not request.future.done():
                    request.future.set_result(result)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)

            # Set exception for all requests
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)

    async def _process_batch(self, batch: list) -> list:
        """
        Process a batch of requests.

        This is a placeholder - in production, this would call the actual
        model inference.

        Args:
            batch: List of BatchRequest objects

        Returns:
            List of results
        """
        # Simulate inference
        await asyncio.sleep(0.001)

        # Return dummy results
        return [{"result": f"processed_{req.request_id}"} for req in batch]

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self._queue)
```

### 3.3 Create the FastAPI Server

The full server implementation is in `src/serving/server.py` (already created). Key points:

- **Async design**: Uses FastAPI async endpoints for concurrent request handling
- **Health checks**: `/health` endpoint for Kubernetes liveness/readiness probes
- **Metrics**: Prometheus metrics on `/metrics` endpoint
- **Request validation**: Pydantic models for type-safe request/response
- **Error handling**: Comprehensive exception handling with proper HTTP status codes

**Test the server**:

```bash
# Start the server
cd ~/model-serving-project
python -m src.serving.server

# In another terminal, test endpoints
curl http://localhost:8000/health

# Test prediction (will fail without model loaded)
curl -X POST http://localhost:8000/v1/predict \
    -H "Content-Type: application/json" \
    -d '{
        "model": "resnet50-fp16",
        "inputs": {"image": "base64_data"},
        "parameters": {}
    }'
```

---

## Step 4: Adding vLLM Support

vLLM enables efficient serving of large language models using PagedAttention.

### 4.1 Configure vLLM

Create `src/llm/config.py`:

```python
"""
LLM Configuration

Configuration classes for vLLM and other LLM serving frameworks.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class LLMConfig:
    """
    Configuration for vLLM engine.

    See vLLM documentation for detailed parameter descriptions:
    https://docs.vllm.ai/en/latest/
    """

    # Model configuration
    model: str = "meta-llama/Llama-2-7b-hf"
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = False
    download_dir: Optional[str] = None

    # Parallelism
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # Memory management
    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = None
    block_size: int = 16
    swap_space: int = 4  # GiB

    # Performance
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    max_paddings: int = 256

    # Sampling defaults
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 50

    # Features
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True

    # Quantization
    quantization: Optional[str] = None  # "awq", "gptq", "squeezellm", etc.

    # Optimization
    enforce_eager: bool = False
    max_context_len_to_capture: int = 8192

    def __post_init__(self):
        """Validate configuration."""
        if self.gpu_memory_utilization <= 0 or self.gpu_memory_utilization > 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")

        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be at least 1")

    def to_engine_args(self) -> Dict[str, Any]:
        """Convert to vLLM AsyncEngineArgs format."""
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "tokenizer_mode": self.tokenizer_mode,
            "trust_remote_code": self.trust_remote_code,
            "download_dir": self.download_dir,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "block_size": self.block_size,
            "swap_space": self.swap_space,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "max_paddings": self.max_paddings,
            "quantization": self.quantization,
            "enforce_eager": self.enforce_eager,
            "max_context_len_to_capture": self.max_context_len_to_capture,
        }


# Preset configurations for common models
LLAMA2_7B_CONFIG = LLMConfig(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,
    max_num_seqs=256,
    gpu_memory_utilization=0.90,
)

LLAMA2_13B_CONFIG = LLMConfig(
    model="meta-llama/Llama-2-13b-hf",
    tensor_parallel_size=2,
    max_num_seqs=128,
    gpu_memory_utilization=0.95,
)

MISTRAL_7B_CONFIG = LLMConfig(
    model="mistralai/Mistral-7B-v0.1",
    tensor_parallel_size=1,
    max_num_seqs=256,
    gpu_memory_utilization=0.90,
    max_model_len=4096,
)
```

The vLLM server implementation is already complete in `src/llm/vllm_server.py`.

### 4.2 Test vLLM Server

Create a test script `scripts/test_vllm.py`:

```python
#!/usr/bin/env python3
"""Test vLLM server functionality."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.config import LLAMA2_7B_CONFIG
from llm.vllm_server import VLLMServer


async def main():
    # Create server with Llama 2 7B config
    server = VLLMServer(LLAMA2_7B_CONFIG)

    # Initialize
    print("Initializing vLLM server...")
    await server.initialize()

    # Generate text
    print("\n=== Non-streaming Generation ===")
    response = await server.generate(
        prompt="Explain machine learning in simple terms:",
        max_tokens=100,
        temperature=0.7,
    )
    print(f"Generated ({response.tokens_generated} tokens in {response.latency:.2f}s):")
    print(response.text)

    # Streaming generation
    print("\n=== Streaming Generation ===")
    print("Generated: ", end="", flush=True)
    async for chunk in server.generate_stream(
        prompt="Write a haiku about AI:",
        max_tokens=50,
    ):
        print(chunk, end="", flush=True)
    print()

    # Get stats
    stats = await server.get_stats()
    print(f"\n=== Server Stats ===")
    print(f"Total requests: {stats.total_requests}")
    print(f"Average latency: {stats.average_latency:.2f}s")
    print(f"Average tokens/sec: {stats.average_tokens_per_second:.1f}")

    # Shutdown
    await server.shutdown()
    print("\nServer shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
```

Run the test:

```bash
python scripts/test_vllm.py
```

**Expected Output**:
```
Initializing vLLM server...
INFO:vllm_server:vLLM engine initialized successfully for model: meta-llama/Llama-2-7b-hf

=== Non-streaming Generation ===
Generated (87 tokens in 1.85s):
Machine learning is a subset of artificial intelligence that enables computers
to learn from data without being explicitly programmed...

=== Streaming Generation ===
Generated: Silicon dreams arise
Neural pathways intertwine
Thinking without thought

=== Server Stats ===
Total requests: 2
Average latency: 1.42s
Average tokens/sec: 52.3

Server shutdown complete
```

**Note**: First run will download the model (7-13 GB), which may take 10-30 minutes.

---

## Step 5: Setting Up Routing

The routing layer distributes requests across multiple backend endpoints.

### 5.1 Implement A/B Testing

Create `src/routing/ab_testing.py`:

```python
"""
A/B Testing Module

Implements statistical A/B testing for model comparison with automatic
winner selection based on configurable metrics.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class VariantStats:
    """Statistics for an A/B test variant."""
    name: str
    requests: int = 0
    successes: int = 0
    failures: int = 0
    total_latency: float = 0.0
    latencies: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.requests == 0:
            return 0.0
        return self.successes / self.requests

    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        if self.successes == 0:
            return 0.0
        return self.total_latency / self.successes

    def record_success(self, latency: float) -> None:
        """Record a successful request."""
        self.requests += 1
        self.successes += 1
        self.total_latency += latency
        self.latencies.append(latency)

    def record_failure(self) -> None:
        """Record a failed request."""
        self.requests += 1
        self.failures += 1


@dataclass
class ABTest:
    """
    A/B test configuration and state.

    Attributes:
        name: Test identifier
        variant_a: Name of variant A (control)
        variant_b: Name of variant B (treatment)
        traffic_split: Percentage of traffic to variant B (0.0-1.0)
        min_samples: Minimum samples before statistical test
        confidence_level: Required confidence level (0.95 = 95%)
        metric: Metric to optimize ('success_rate' or 'latency')
    """

    name: str
    variant_a: str
    variant_b: str
    traffic_split: float = 0.5
    min_samples: int = 100
    confidence_level: float = 0.95
    metric: str = "success_rate"  # or "latency"

    stats_a: VariantStats = field(init=False)
    stats_b: VariantStats = field(init=False)
    start_time: float = field(default_factory=time.time)
    winner: Optional[str] = None

    def __post_init__(self):
        """Initialize variant stats."""
        self.stats_a = VariantStats(name=self.variant_a)
        self.stats_b = VariantStats(name=self.variant_b)

        if not 0 <= self.traffic_split <= 1:
            raise ValueError("traffic_split must be between 0 and 1")

        logger.info(
            f"ABTest '{self.name}' initialized: "
            f"{self.variant_a} vs {self.variant_b} "
            f"(split={self.traffic_split:.0%})"
        )

    def select_variant(self, user_id: Optional[str] = None) -> str:
        """
        Select a variant for the request.

        Args:
            user_id: Optional user ID for consistent assignment

        Returns:
            Selected variant name
        """
        if self.winner:
            return self.winner

        # Use consistent hashing if user_id provided
        if user_id:
            import hashlib
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            threshold = hash_val % 100 / 100.0
        else:
            threshold = np.random.random()

        return self.variant_b if threshold < self.traffic_split else self.variant_a

    def record_result(
        self,
        variant: str,
        success: bool,
        latency: Optional[float] = None
    ) -> None:
        """
        Record a result for a variant.

        Args:
            variant: Variant name
            success: Whether request succeeded
            latency: Request latency (if successful)
        """
        stats = self.stats_a if variant == self.variant_a else self.stats_b

        if success and latency is not None:
            stats.record_success(latency)
        else:
            stats.record_failure()

    def analyze(self) -> Dict[str, any]:
        """
        Perform statistical analysis of A/B test results.

        Returns:
            Analysis results including p-value and recommendation
        """
        # Check minimum samples
        if (self.stats_a.requests < self.min_samples or
            self.stats_b.requests < self.min_samples):
            return {
                "status": "insufficient_data",
                "message": f"Need {self.min_samples} samples per variant",
                "samples_a": self.stats_a.requests,
                "samples_b": self.stats_b.requests,
            }

        if self.metric == "success_rate":
            return self._analyze_success_rate()
        elif self.metric == "latency":
            return self._analyze_latency()
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _analyze_success_rate(self) -> Dict[str, any]:
        """Analyze using success rate (proportion test)."""
        # Two-proportion z-test
        n1, n2 = self.stats_a.requests, self.stats_b.requests
        x1, x2 = self.stats_a.successes, self.stats_b.successes

        p1 = x1 / n1
        p2 = x2 / n2

        # Pooled proportion
        p_pool = (x1 + x2) / (n1 + n2)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

        # Z-statistic
        z = (p2 - p1) / se if se > 0 else 0

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Determine winner
        significant = p_value < (1 - self.confidence_level)
        if significant:
            winner = self.variant_b if p2 > p1 else self.variant_a
            improvement = abs((p2 - p1) / p1) * 100
        else:
            winner = None
            improvement = 0

        return {
            "status": "complete",
            "metric": "success_rate",
            "variant_a": {
                "success_rate": p1,
                "samples": n1,
            },
            "variant_b": {
                "success_rate": p2,
                "samples": n2,
            },
            "p_value": p_value,
            "significant": significant,
            "winner": winner,
            "improvement_pct": improvement,
            "recommendation": (
                f"Deploy {winner} - {improvement:.1f}% improvement"
                if winner else
                "No significant difference - keep testing"
            ),
        }

    def _analyze_latency(self) -> Dict[str, any]:
        """Analyze using latency (t-test)."""
        latencies_a = np.array(self.stats_a.latencies)
        latencies_b = np.array(self.stats_b.latencies)

        if len(latencies_a) < 2 or len(latencies_b) < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 successful requests per variant",
            }

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(latencies_a, latencies_b)

        mean_a = np.mean(latencies_a)
        mean_b = np.mean(latencies_b)

        # Determine winner (lower latency is better)
        significant = p_value < (1 - self.confidence_level)
        if significant:
            winner = self.variant_b if mean_b < mean_a else self.variant_a
            improvement = abs((mean_b - mean_a) / mean_a) * 100
        else:
            winner = None
            improvement = 0

        return {
            "status": "complete",
            "metric": "latency",
            "variant_a": {
                "mean_latency_ms": mean_a * 1000,
                "samples": len(latencies_a),
            },
            "variant_b": {
                "mean_latency_ms": mean_b * 1000,
                "samples": len(latencies_b),
            },
            "p_value": p_value,
            "significant": significant,
            "winner": winner,
            "improvement_pct": improvement,
            "recommendation": (
                f"Deploy {winner} - {improvement:.1f}% faster"
                if winner else
                "No significant difference - keep testing"
            ),
        }


class ABTestManager:
    """Manager for multiple concurrent A/B tests."""

    def __init__(self):
        """Initialize A/B test manager."""
        self.tests: Dict[str, ABTest] = {}
        logger.info("ABTestManager initialized")

    def create_test(self, test: ABTest) -> None:
        """Create a new A/B test."""
        if test.name in self.tests:
            raise ValueError(f"Test '{test.name}' already exists")

        self.tests[test.name] = test
        logger.info(f"Created A/B test: {test.name}")

    def get_test(self, name: str) -> Optional[ABTest]:
        """Get an A/B test by name."""
        return self.tests.get(name)

    def delete_test(self, name: str) -> bool:
        """Delete an A/B test."""
        if name in self.tests:
            del self.tests[name]
            logger.info(f"Deleted A/B test: {name}")
            return True
        return False

    def list_tests(self) -> List[str]:
        """List all active tests."""
        return list(self.tests.keys())
```

### 5.2 Implement Canary Deployments

Create `src/routing/canary.py`:

```python
"""
Canary Deployment Module

Implements progressive canary rollouts with automatic rollback based on
error rates and latency metrics.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class CanaryState(Enum):
    """Canary deployment states."""
    INITIALIZING = "initializing"
    RAMPING = "ramping"
    STABLE = "stable"
    ROLLING_BACK = "rolling_back"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CanaryMetrics:
    """Metrics for canary deployment monitoring."""
    requests: int = 0
    successes: int = 0
    failures: int = 0
    total_latency: float = 0.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.requests == 0:
            return 0.0
        return self.failures / self.requests

    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        if self.successes == 0:
            return 0.0
        return self.total_latency / self.successes

    def record_success(self, latency: float) -> None:
        """Record successful request."""
        self.requests += 1
        self.successes += 1
        self.total_latency += latency

    def record_failure(self) -> None:
        """Record failed request."""
        self.requests += 1
        self.failures += 1


@dataclass
class CanaryConfig:
    """
    Configuration for canary deployment.

    Attributes:
        name: Deployment identifier
        baseline_version: Current stable version
        canary_version: New version being tested
        initial_traffic_pct: Starting traffic percentage
        increment_pct: Traffic increment per stage
        increment_interval_s: Seconds between increments
        max_error_rate: Maximum acceptable error rate
        max_latency_increase: Maximum latency increase (ratio)
        rollback_on_failure: Auto-rollback on threshold breach
    """

    name: str
    baseline_version: str
    canary_version: str
    initial_traffic_pct: float = 5.0
    increment_pct: float = 10.0
    increment_interval_s: float = 300.0  # 5 minutes
    max_error_rate: float = 0.05  # 5%
    max_latency_increase: float = 1.5  # 50% increase
    min_requests_per_stage: int = 100
    rollback_on_failure: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.initial_traffic_pct <= 100:
            raise ValueError("initial_traffic_pct must be between 0 and 100")
        if not 0 < self.increment_pct <= 100:
            raise ValueError("increment_pct must be between 0 and 100")


class CanaryDeployment:
    """
    Progressive canary deployment with automatic rollback.

    Gradually increases traffic to a new version while monitoring error rates
    and latency. Automatically rolls back if thresholds are exceeded.
    """

    def __init__(self, config: CanaryConfig):
        """
        Initialize canary deployment.

        Args:
            config: Canary configuration
        """
        self.config = config
        self.state = CanaryState.INITIALIZING
        self.current_traffic_pct = 0.0

        self.baseline_metrics = CanaryMetrics()
        self.canary_metrics = CanaryMetrics()

        self.start_time: Optional[float] = None
        self.last_increment_time: Optional[float] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        logger.info(
            f"CanaryDeployment '{config.name}' initialized: "
            f"{config.baseline_version} -> {config.canary_version}"
        )

    async def start(self) -> None:
        """Start the canary deployment."""
        if self.state != CanaryState.INITIALIZING:
            logger.warning(f"Deployment already started (state={self.state})")
            return

        self.state = CanaryState.RAMPING
        self.current_traffic_pct = self.config.initial_traffic_pct
        self.start_time = time.time()
        self.last_increment_time = time.time()

        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info(
            f"Canary deployment started: {self.current_traffic_pct:.1f}% traffic "
            f"to {self.config.canary_version}"
        )

    async def stop(self) -> None:
        """Stop the canary deployment."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Canary deployment stopped (final state={self.state})")

    def select_version(self) -> str:
        """
        Select version for request based on current traffic split.

        Returns:
            Selected version name
        """
        if self.state in [CanaryState.ROLLING_BACK, CanaryState.FAILED]:
            return self.config.baseline_version

        if self.state == CanaryState.COMPLETED:
            return self.config.canary_version

        # Random selection based on traffic percentage
        import random
        if random.random() * 100 < self.current_traffic_pct:
            return self.config.canary_version
        return self.config.baseline_version

    def record_result(
        self,
        version: str,
        success: bool,
        latency: Optional[float] = None
    ) -> None:
        """
        Record request result for version.

        Args:
            version: Version name
            success: Whether request succeeded
            latency: Request latency (if successful)
        """
        metrics = (
            self.canary_metrics
            if version == self.config.canary_version
            else self.baseline_metrics
        )

        if success and latency is not None:
            metrics.record_success(latency)
        else:
            metrics.record_failure()

    async def _monitor_loop(self) -> None:
        """Background monitoring and traffic ramping."""
        logger.info("Canary monitoring loop started")

        while self.state == CanaryState.RAMPING:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                async with self._lock:
                    # Check health
                    if not self._check_health():
                        if self.config.rollback_on_failure:
                            await self._rollback()
                        else:
                            self.state = CanaryState.FAILED
                        break

                    # Check if ready to increment
                    if self._should_increment_traffic():
                        await self._increment_traffic()

                    # Check if deployment complete
                    if self.current_traffic_pct >= 100:
                        self.state = CanaryState.COMPLETED
                        logger.info(
                            f"Canary deployment completed successfully: "
                            f"{self.config.canary_version} at 100%"
                        )
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}", exc_info=True)

        logger.info("Canary monitoring loop stopped")

    def _check_health(self) -> bool:
        """
        Check canary health against thresholds.

        Returns:
            True if healthy, False if thresholds exceeded
        """
        # Need minimum requests
        if self.canary_metrics.requests < self.config.min_requests_per_stage:
            return True  # Not enough data yet

        # Check error rate
        if self.canary_metrics.error_rate > self.config.max_error_rate:
            logger.error(
                f"Canary error rate ({self.canary_metrics.error_rate:.2%}) "
                f"exceeds threshold ({self.config.max_error_rate:.2%})"
            )
            return False

        # Check latency (if baseline has data)
        if self.baseline_metrics.successes > 0:
            latency_ratio = (
                self.canary_metrics.average_latency /
                self.baseline_metrics.average_latency
            )

            if latency_ratio > self.config.max_latency_increase:
                logger.error(
                    f"Canary latency increase ({latency_ratio:.2f}x) "
                    f"exceeds threshold ({self.config.max_latency_increase:.2f}x)"
                )
                return False

        return True

    def _should_increment_traffic(self) -> bool:
        """Check if ready to increment traffic."""
        if self.last_increment_time is None:
            return False

        # Check time interval
        time_since_increment = time.time() - self.last_increment_time
        if time_since_increment < self.config.increment_interval_s:
            return False

        # Check minimum requests
        if self.canary_metrics.requests < self.config.min_requests_per_stage:
            return False

        return True

    async def _increment_traffic(self) -> None:
        """Increment canary traffic percentage."""
        old_pct = self.current_traffic_pct
        self.current_traffic_pct = min(
            100.0,
            self.current_traffic_pct + self.config.increment_pct
        )
        self.last_increment_time = time.time()

        # Reset metrics for next stage
        self.canary_metrics = CanaryMetrics()
        self.baseline_metrics = CanaryMetrics()

        logger.info(
            f"Incremented canary traffic: {old_pct:.1f}% -> "
            f"{self.current_traffic_pct:.1f}%"
        )

    async def _rollback(self) -> None:
        """Roll back to baseline version."""
        logger.warning(
            f"Rolling back canary deployment: {self.config.canary_version} -> "
            f"{self.config.baseline_version}"
        )

        self.state = CanaryState.ROLLING_BACK
        self.current_traffic_pct = 0.0

        # Give time for in-flight requests
        await asyncio.sleep(5)

        self.state = CanaryState.FAILED
        logger.info("Rollback complete")

    def get_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "baseline_version": self.config.baseline_version,
            "canary_version": self.config.canary_version,
            "current_traffic_pct": self.current_traffic_pct,
            "baseline_metrics": {
                "requests": self.baseline_metrics.requests,
                "error_rate": self.baseline_metrics.error_rate,
                "average_latency_ms": self.baseline_metrics.average_latency * 1000,
            },
            "canary_metrics": {
                "requests": self.canary_metrics.requests,
                "error_rate": self.canary_metrics.error_rate,
                "average_latency_ms": self.canary_metrics.average_latency * 1000,
            },
            "uptime_s": time.time() - self.start_time if self.start_time else 0,
        }
```

The router implementation is already complete in `src/routing/router.py`.

---

## Step 6: Distributed Tracing Integration

Distributed tracing provides end-to-end visibility into request flows.

### 6.1 Create Tracing Module

Create `src/tracing/tracer.py`:

```python
"""
Distributed Tracing with OpenTelemetry

Implements distributed tracing for model serving with Jaeger export.
"""

import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.trace import Span, Status, StatusCode

logger = logging.getLogger(__name__)


class ModelServingTracer:
    """
    OpenTelemetry tracer for model serving.

    Provides distributed tracing capabilities with Jaeger export.
    """

    def __init__(
        self,
        service_name: str = "model-serving",
        jaeger_host: str = "localhost",
        jaeger_port: int = 6831,
        sample_rate: float = 1.0,
    ):
        """
        Initialize tracer.

        Args:
            service_name: Service name for traces
            jaeger_host: Jaeger agent host
            jaeger_port: Jaeger agent port
            sample_rate: Sampling rate (0.0-1.0)
        """
        self.service_name = service_name
        self.jaeger_host = jaeger_host
        self.jaeger_port = jaeger_port
        self.sample_rate = sample_rate

        self._setup_tracer()

        logger.info(
            f"Tracing initialized: service={service_name}, "
            f"jaeger={jaeger_host}:{jaeger_port}, sample_rate={sample_rate}"
        )

    def _setup_tracer(self) -> None:
        """Set up OpenTelemetry tracer with Jaeger export."""
        # Create resource
        resource = Resource.create({
            SERVICE_NAME: self.service_name
        })

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Create Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=self.jaeger_host,
            agent_port=self.jaeger_port,
        )

        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(span_processor)

        # Set global tracer provider
        trace.set_tracer_provider(provider)

        # Get tracer
        self.tracer = trace.get_tracer(__name__)

    @contextmanager
    def trace_inference(
        self,
        model_name: str,
        request_id: str,
        **attributes
    ):
        """
        Context manager for tracing inference requests.

        Args:
            model_name: Model being invoked
            request_id: Request identifier
            **attributes: Additional span attributes

        Yields:
            Span object
        """
        with self.tracer.start_as_current_span(
            "model.inference",
            attributes={
                "model.name": model_name,
                "request.id": request_id,
                **attributes,
            }
        ) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        **attributes
    ):
        """
        Context manager for tracing generic operations.

        Args:
            operation_name: Operation name
            **attributes: Span attributes

        Yields:
            Span object
        """
        with self.tracer.start_as_current_span(
            operation_name,
            attributes=attributes
        ) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def add_event(self, span: Span, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to a span."""
        span.add_event(name, attributes=attributes or {})

    def set_attribute(self, span: Span, key: str, value: Any) -> None:
        """Set a span attribute."""
        span.set_attribute(key, value)


# Global tracer instance
_tracer: Optional[ModelServingTracer] = None


def initialize_tracing(
    service_name: str = "model-serving",
    jaeger_host: str = "localhost",
    jaeger_port: int = 6831,
    sample_rate: float = 1.0,
) -> ModelServingTracer:
    """
    Initialize global tracer.

    Args:
        service_name: Service name
        jaeger_host: Jaeger host
        jaeger_port: Jaeger port
        sample_rate: Sampling rate

    Returns:
        Tracer instance
    """
    global _tracer
    _tracer = ModelServingTracer(
        service_name=service_name,
        jaeger_host=jaeger_host,
        jaeger_port=jaeger_port,
        sample_rate=sample_rate,
    )
    return _tracer


def get_tracer() -> Optional[ModelServingTracer]:
    """Get global tracer instance."""
    return _tracer
```

### 6.2 Create FastAPI Middleware

Create `src/tracing/middleware.py`:

```python
"""
FastAPI Tracing Middleware

Automatically traces FastAPI requests with OpenTelemetry.
"""

import time
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .tracer import get_tracer


class TracingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic request tracing.

    Traces all HTTP requests with OpenTelemetry, including:
    - Request method and path
    - Response status code
    - Request duration
    - Query parameters and headers (configurable)
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with tracing."""
        tracer = get_tracer()

        if tracer is None:
            # Tracing not initialized, pass through
            return await call_next(request)

        # Extract request info
        method = request.method
        path = request.url.path
        trace_id = getattr(request.state, 'trace_id', 'unknown')

        # Create span
        with tracer.trace_operation(
            f"{method} {path}",
            **{
                "http.method": method,
                "http.route": path,
                "http.url": str(request.url),
                "http.trace_id": trace_id,
            }
        ) as span:
            start_time = time.time()

            # Process request
            response = await call_next(request)

            # Record response info
            duration = time.time() - start_time
            tracer.set_attribute(span, "http.status_code", response.status_code)
            tracer.set_attribute(span, "http.duration_ms", duration * 1000)

            # Add response header with trace ID
            response.headers["X-Trace-ID"] = trace_id

            return response
```

### 6.3 Integrate Tracing into Server

Add tracing to the FastAPI server. Modify `src/serving/server.py` to include:

```python
from tracing import initialize_tracing
from tracing.middleware import TracingMiddleware

# In lifespan startup:
async def lifespan(app: FastAPI):
    # ... existing startup code ...

    # Initialize tracing
    initialize_tracing(
        service_name="model-serving",
        jaeger_host=os.getenv("JAEGER_HOST", "localhost"),
        jaeger_port=int(os.getenv("JAEGER_PORT", "6831")),
        sample_rate=float(os.getenv("TRACE_SAMPLE_RATE", "0.1")),
    )

    yield
    # ... shutdown code ...

# Add middleware
app.add_middleware(TracingMiddleware)
```

---

## Step 7: Kubernetes Deployment

Now we'll deploy to Kubernetes with auto-scaling and GPU support.

### 7.1 Create Docker Image

Create `docker/Dockerfile`:

```dockerfile
# Multi-stage build for model serving

# Stage 1: Base image with CUDA
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Stage 2: Install Python dependencies
FROM base AS dependencies

WORKDIR /tmp

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies AS application

WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY .env.example ./.env

# Create model directory
RUN mkdir -p /models /tmp/model_cache

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python3", "-m", "uvicorn", "src.serving.server:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

Create `docker/docker-compose.yml` for local testing:

```yaml
version: '3.8'

services:
  model-serving:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
      - "9090:9090"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_CACHE_DIR=/models
      - JAEGER_ENABLED=true
      - JAEGER_HOST=jaeger
    volumes:
      - ../models:/models
      - model-cache:/tmp/model_cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - jaeger
      - prometheus

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "5775:5775/udp"
      - "6831:6831/udp"
      - "6832:6832/udp"
      - "5778:5778"
      - "16686:16686"
      - "14268:14268"
      - "14250:14250"
      - "9411:9411"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ../monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ../monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  model-cache:
  prometheus-data:
  grafana-data:
```

Build and test:

```bash
cd ~/model-serving-project

# Build image
docker build -t model-serving:latest -f docker/Dockerfile .

# Run with Docker Compose
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f model-serving

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:9091  # Prometheus
curl http://localhost:16686  # Jaeger UI
curl http://localhost:3000   # Grafana (admin/admin)
```

### 7.2 Create Kubernetes Manifests

Create `kubernetes/base/namespace.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: model-serving
  labels:
    name: model-serving
```

Create `kubernetes/base/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
  namespace: model-serving
  labels:
    app: model-serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-serving
  template:
    metadata:
      labels:
        app: model-serving
    spec:
      containers:
      - name: model-serving
        image: model-serving:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: MODEL_CACHE_DIR
          value: "/models"
        - name: JAEGER_HOST
          value: "jaeger.monitoring.svc.cluster.local"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: model-cache
          mountPath: /models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      nodeSelector:
        accelerator: nvidia-gpu
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

Create `kubernetes/base/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-serving
  namespace: model-serving
  labels:
    app: model-serving
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  - port: 9090
    targetPort: 9090
    protocol: TCP
    name: metrics
  selector:
    app: model-serving
```

Create `kubernetes/base/hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-serving-hpa
  namespace: model-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-serving
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
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

Create `kubernetes/base/pvc.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
  namespace: model-serving
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

Create `kubernetes/base/kustomization.yaml`:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: model-serving

resources:
  - namespace.yaml
  - deployment.yaml
  - service.yaml
  - hpa.yaml
  - pvc.yaml

commonLabels:
  project: model-serving
  managed-by: kustomize
```

### 7.3 Create Environment Overlays

Create `kubernetes/overlays/prod/kustomization.yaml`:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

bases:
  - ../../base

namePrefix: prod-

replicas:
  - name: model-serving
    count: 5

images:
  - name: model-serving
    newName: your-registry.com/model-serving
    newTag: v1.0.0

patchesStrategicMerge:
  - deployment-patch.yaml
```

Create `kubernetes/overlays/prod/deployment-patch.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
spec:
  template:
    spec:
      containers:
      - name: model-serving
        env:
        - name: LOG_LEVEL
          value: "WARNING"
        - name: METRICS_ENABLED
          value: "true"
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
```

### 7.4 Deploy to Kubernetes

```bash
# Build and push image
docker build -t your-registry.com/model-serving:v1.0.0 -f docker/Dockerfile .
docker push your-registry.com/model-serving:v1.0.0

# Deploy to production
kubectl apply -k kubernetes/overlays/prod/

# Verify deployment
kubectl get pods -n model-serving
kubectl get svc -n model-serving
kubectl get hpa -n model-serving

# View logs
kubectl logs -n model-serving -l app=model-serving -f

# Port forward for testing
kubectl port-forward -n model-serving svc/model-serving 8000:80

# Test
curl http://localhost:8000/health
```

---

## Step 8: Monitoring and Observability

### 8.1 Configure Prometheus

Create `monitoring/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'model-serving'

scrape_configs:
  - job_name: 'model-serving'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - model-serving
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: model-serving
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: pod
      - source_labels: [__address__]
        action: replace
        regex: ([^:]+):.*
        replacement: ${1}:9090
        target_label: __address__

  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

rule_files:
  - /etc/prometheus/alerts/*.yml
```

Create `monitoring/prometheus/alerts/model-serving.yml`:

```yaml
groups:
  - name: model_serving_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(model_serving_requests_total{status="error"}[5m]))
          /
          sum(rate(model_serving_requests_total[5m]))
          > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.99,
            sum(rate(model_serving_request_duration_seconds_bucket[5m])) by (le, model)
          ) > 1.0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High P99 latency for model {{ $labels.model }}"
          description: "P99 latency is {{ $value }}s (threshold: 1s)"

      - alert: GPUMemoryHigh
        expr: |
          nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory usage critical"
          description: "GPU {{ $labels.gpu }} memory at {{ $value | humanizePercentage }}"

      - alert: ServiceDown
        expr: up{job="model-serving"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Model serving instance down"
          description: "Instance {{ $labels.instance }} has been down for >2 minutes"
```

### 8.2 Create Grafana Dashboards

Create `monitoring/grafana/dashboards/model-serving.json` (abbreviated):

```json
{
  "dashboard": {
    "title": "Model Serving Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "sum(rate(model_serving_requests_total[5m])) by (model)"
          }
        ]
      },
      {
        "title": "P50/P95/P99 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum(rate(model_serving_request_duration_seconds_bucket[5m])) by (le, model))",
            "legendFormat": "P50 - {{model}}"
          },
          {
            "expr": "histogram_quantile(0.95, sum(rate(model_serving_request_duration_seconds_bucket[5m])) by (le, model))",
            "legendFormat": "P95 - {{model}}"
          },
          {
            "expr": "histogram_quantile(0.99, sum(rate(model_serving_request_duration_seconds_bucket[5m])) by (le, model))",
            "legendFormat": "P99 - {{model}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "sum(rate(model_serving_requests_total{status='error'}[5m])) / sum(rate(model_serving_requests_total[5m]))"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization"
          }
        ]
      }
    ]
  }
}
```

---

## Step 9: Production Hardening

### 9.1 Add Rate Limiting

Install `slowapi`:

```bash
pip install slowapi
```

Add to `src/serving/server.py`:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/v1/predict")
@limiter.limit("100/minute")
async def predict(request: Request, body: PredictRequest):
    # ... existing code ...
```

### 9.2 Add Authentication

Create `src/serving/auth.py`:

```python
"""
Authentication and Authorization

Implements API key and JWT-based authentication.
"""

from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

security = HTTPBearer()

API_KEYS = set(os.getenv("API_KEYS", "").split(","))


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    """
    Verify API key from Authorization header.

    Args:
        credentials: HTTP bearer credentials

    Returns:
        API key if valid

    Raises:
        HTTPException: If invalid or missing
    """
    if credentials.credentials not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
```

Add to endpoints:

```python
from .auth import verify_api_key

@app.post("/v1/predict")
async def predict(
    request: Request,
    body: PredictRequest,
    api_key: str = Depends(verify_api_key)
):
    # ... existing code ...
```

### 9.3 Add Input Validation

Enhance Pydantic models with validators:

```python
from pydantic import BaseModel, Field, validator
import base64

class PredictRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=100)
    inputs: Dict[str, Any]

    @validator('inputs')
    def validate_inputs(cls, v):
        """Validate input data."""
        if not v:
            raise ValueError("inputs cannot be empty")

        # Validate image data if present
        if 'image' in v:
            try:
                # Verify base64 encoding
                base64.b64decode(v['image'])
            except Exception:
                raise ValueError("Invalid base64 image data")

        return v
```

### 9.4 Add Logging

Configure structured logging:

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
```

---

## Common Pitfalls and Debugging

### Issue 1: CUDA Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:
1. Reduce batch size
2. Lower model precision (FP16 or INT8)
3. Reduce `gpu_memory_utilization` for vLLM
4. Clear GPU cache: `torch.cuda.empty_cache()`

**Debug Commands**:
```bash
# Monitor GPU memory
nvidia-smi -l 1

# Check TensorRT memory usage
CUDA_VISIBLE_DEVICES=0 python -c "import tensorrt as trt; print(trt.get_plugin_registry())"
```

### Issue 2: Slow Model Loading

**Symptoms**:
- First request takes >30 seconds
- Cold start latency high

**Solutions**:
1. Implement model warmup
2. Use model caching
3. Reduce TensorRT timing iterations
4. Use pre-built engines

**Warmup Code**:
```python
async def warmup_model(model_name: str, iterations: int = 10):
    """Warmup model with dummy requests."""
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    for i in range(iterations):
        await _run_inference(model_name, {"input": dummy_input}, {})

    logger.info(f"Model {model_name} warmed up")
```

### Issue 3: High P99 Latency

**Symptoms**:
- P50 latency good, P99 >1s
- Sporadic slow requests

**Solutions**:
1. Tune batch processor timeout
2. Increase worker count
3. Add request queuing
4. Check GPU thermal throttling

**Debug**:
```bash
# Check GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv

# Monitor request latencies
curl http://localhost:9090/metrics | grep duration_seconds
```

### Issue 4: Kubernetes Pod Crashes

**Symptoms**:
```
CrashLoopBackOff
OOMKilled
```

**Solutions**:
1. Increase memory limits
2. Check resource requests/limits ratio
3. Verify GPU drivers installed
4. Check liveness/readiness probes

**Debug**:
```bash
# Check pod events
kubectl describe pod -n model-serving <pod-name>

# Check logs
kubectl logs -n model-serving <pod-name> --previous

# Check resources
kubectl top pod -n model-serving
```

---

## Testing and Validation

### Run Unit Tests

```bash
pytest tests/ -v --cov=src --cov-report=html

# Run specific test
pytest tests/test_tensorrt.py::test_model_conversion -v

# Run with markers
pytest -m "not slow" tests/
```

### Run Integration Tests

```bash
# Start services
docker-compose -f docker/docker-compose.yml up -d

# Run integration tests
pytest tests/integration/ -v --slow

# Load testing
python benchmarks/load_test.py --url http://localhost:8000 --requests 10000
```

### Benchmark Performance

```bash
# Latency test
python benchmarks/latency_test.py \
    --model resnet50-fp16 \
    --batch-sizes 1,4,8,16,32 \
    --iterations 1000

# Throughput test
python benchmarks/throughput_test.py \
    --model resnet50-fp16 \
    --duration 60 \
    --concurrent-requests 100
```

---

## Next Steps

Congratulations! You've built a production-ready high-performance model serving system. Next steps:

1. **Multi-Model Serving**: Add support for serving multiple models simultaneously
2. **Model Versioning**: Implement model version management and rollback
3. **Advanced Batching**: Add adaptive batching with dynamic timeout adjustment
4. **Cost Optimization**: Implement spot instance support and cost monitoring
5. **Security**: Add mTLS, network policies, and secret management
6. **Observability**: Add custom metrics, APM integration, and log aggregation
7. **CI/CD**: Implement automated testing and deployment pipelines
8. **Disaster Recovery**: Add backup/restore and multi-region deployment

For more information, see:
- [API Reference](API.md)
- [Architecture Documentation](ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Operations Runbook](RUNBOOK.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
