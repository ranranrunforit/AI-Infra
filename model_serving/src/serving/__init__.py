"""
Model Serving Module

High-performance serving infrastructure for ML models with:
- FastAPI-based async server
- Multi-format model loading (TensorRT, PyTorch, ONNX)
- Dynamic batching for throughput optimization
- Model lifecycle management
- Metrics and observability

Example:
    >>> from serving import ModelLoader, DynamicBatchProcessor
    >>> loader = ModelLoader(cache_dir="/tmp/models")
    >>> loader.load_model("resnet50-fp16", model_format="tensorrt")
"""

from .model_loader import (
    ModelLoader,
    ModelFormat,
    ModelInfo,
    TensorRTModel,
)

from .batch_processor import (
    DynamicBatchProcessor,
    BatchRequest,
    RequestPriority,
    BatchStats,
)

from .warmup import ModelWarmup

# Server components are imported separately to avoid FastAPI dependency issues
# from .server import app, PredictRequest, PredictResponse

__all__ = [
    # Model Loader
    "ModelLoader",
    "ModelFormat",
    "ModelInfo",
    "TensorRTModel",
    # Batch Processor
    "DynamicBatchProcessor",
    "BatchRequest",
    "RequestPriority",
    "BatchStats",
    # Warmup
    "ModelWarmup",
]

__version__ = "1.0.0"
