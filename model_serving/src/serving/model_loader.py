"""
Model Loader and Cache Manager

Handles loading, caching, and lifecycle management of ML models in multiple formats:
- TensorRT engines (.trt, .plan)
- PyTorch models (.pt, .pth)
- ONNX models (.onnx)

Features:
- Thread-safe model management
- LRU cache with memory limits
- Model warmup for consistent performance
- Resource cleanup and GPU memory management
- Model versioning support
"""

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """Supported model formats."""
    TENSORRT = "tensorrt"
    PYTORCH = "pytorch"
    ONNX = "onnx"


@dataclass
class ModelInfo:
    """Metadata for a loaded model."""
    name: str
    format: ModelFormat
    path: Optional[Path]
    memory_mb: float
    warmup_completed: bool = False
    load_time_seconds: float = 0.0
    version: str = "1.0"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TensorRTModel:
    """Wrapper for TensorRT engine execution."""

    def __init__(self, engine: trt.ICudaEngine):
        """
        Initialize TensorRT model wrapper.

        Args:
            engine: TensorRT engine
        """
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate buffers
        self.inputs = {}
        self.outputs = {}
        self.bindings = []

        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        """Allocate GPU buffers for all inputs and outputs."""
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = trt.volume(shape)

            # Allocate device memory
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.inputs[binding_name] = {
                    "index": i,
                    "shape": shape,
                    "dtype": dtype,
                    "device": device_mem,
                }
            else:
                self.outputs[binding_name] = {
                    "index": i,
                    "shape": shape,
                    "dtype": dtype,
                    "device": device_mem,
                }

        logger.info(
            f"Allocated buffers: {len(self.inputs)} inputs, {len(self.outputs)} outputs"
        )

    def infer(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference on the TensorRT engine.

        Args:
            input_data: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        # Copy inputs to device
        for name, data in input_data.items():
            if name in self.inputs:
                input_info = self.inputs[name]
                host_mem = np.ascontiguousarray(data).astype(input_info["dtype"])
                cuda.memcpy_htod_async(
                    input_info["device"],
                    host_mem,
                    self.stream
                )

        # Execute inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Copy outputs to host
        results = {}
        for name, output_info in self.outputs.items():
            shape = output_info["shape"]
            dtype = output_info["dtype"]

            # Handle dynamic shapes
            if -1 in shape:
                shape = self.context.get_binding_shape(output_info["index"])

            size = trt.volume(shape)
            host_mem = np.empty(size, dtype=dtype)

            cuda.memcpy_dtoh_async(
                host_mem,
                output_info["device"],
                self.stream
            )

            results[name] = host_mem.reshape(shape)

        # Wait for completion
        self.stream.synchronize()

        return results

    def __del__(self):
        """Clean up GPU memory."""
        for input_info in self.inputs.values():
            input_info["device"].free()
        for output_info in self.outputs.values():
            output_info["device"].free()


class ModelLoader:
    """
    Thread-safe model loader with caching and lifecycle management.

    Manages loading, caching, and unloading of models in multiple formats.
    Implements LRU cache with configurable memory limits.

    Example:
        >>> loader = ModelLoader(cache_dir="/tmp/models", max_cache_size_mb=4096)
        >>> loader.load_model("resnet50-fp16", model_format="tensorrt")
        >>> model = loader.get_model("resnet50-fp16")
        >>> predictions = model.infer(inputs)
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "/tmp/model_cache",
        max_cache_size_mb: int = 4096,
        enable_warmup: bool = True,
    ):
        """
        Initialize model loader.

        Args:
            cache_dir: Directory for caching models
            max_cache_size_mb: Maximum cache size in MB
            enable_warmup: Whether to warmup models after loading
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_cache_size_mb = max_cache_size_mb
        self.enable_warmup = enable_warmup

        # Model cache (OrderedDict for LRU behavior)
        self._models: OrderedDict[str, Any] = OrderedDict()
        self._model_info: Dict[str, ModelInfo] = {}

        # Thread safety
        self._lock = threading.RLock()

        # TensorRT runtime (shared across models)
        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        self._trt_runtime = trt.Runtime(self._trt_logger)

        logger.info(
            f"Initialized ModelLoader: cache_dir={cache_dir}, "
            f"max_cache_size={max_cache_size_mb}MB"
        )

    def load_model(
        self,
        model_name: str,
        model_format: Union[str, ModelFormat] = ModelFormat.TENSORRT,
        model_path: Optional[Union[str, Path]] = None,
        force_reload: bool = False,
    ) -> None:
        """
        Load a model into the cache.

        Args:
            model_name: Unique identifier for the model
            model_format: Model format (tensorrt, pytorch, onnx)
            model_path: Path to model file (if None, searches in cache_dir)
            force_reload: Force reload even if already cached

        Raises:
            FileNotFoundError: If model file not found
            RuntimeError: If model loading fails
        """
        with self._lock:
            # Convert format to enum
            if isinstance(model_format, str):
                model_format = ModelFormat(model_format.lower())

            # Check if already loaded
            if model_name in self._models and not force_reload:
                logger.info(f"Model {model_name} already loaded, using cached version")
                # Move to end (most recently used)
                self._models.move_to_end(model_name)
                return

            # Find model file
            if model_path is None:
                model_path = self._find_model_file(model_name, model_format)
            else:
                model_path = Path(model_path)

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            logger.info(f"Loading model {model_name} from {model_path}")

            import time
            load_start = time.time()

            # Load model based on format
            if model_format == ModelFormat.TENSORRT:
                model = self._load_tensorrt_model(model_path)
            elif model_format == ModelFormat.PYTORCH:
                model = self._load_pytorch_model(model_path)
            elif model_format == ModelFormat.ONNX:
                model = self._load_onnx_model(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_format}")

            load_time = time.time() - load_start

            # Estimate memory usage
            memory_mb = self._estimate_model_memory(model, model_format)

            # Check cache size and evict if necessary
            self._evict_if_needed(memory_mb)

            # Add to cache
            self._models[model_name] = model

            # Store metadata
            self._model_info[model_name] = ModelInfo(
                name=model_name,
                format=model_format,
                path=model_path,
                memory_mb=memory_mb,
                load_time_seconds=load_time,
            )

            logger.info(
                f"Model {model_name} loaded successfully "
                f"({memory_mb:.1f}MB, {load_time:.2f}s)"
            )

            # Warmup model if enabled
            if self.enable_warmup:
                self._warmup_model(model_name)

    def _find_model_file(
        self,
        model_name: str,
        model_format: ModelFormat
    ) -> Path:
        """
        Find model file in cache directory.

        Args:
            model_name: Model name
            model_format: Model format

        Returns:
            Path to model file

        Raises:
            FileNotFoundError: If model not found
        """
        # Common extensions for each format
        extensions = {
            ModelFormat.TENSORRT: [".trt", ".plan", ".engine"],
            ModelFormat.PYTORCH: [".pt", ".pth"],
            ModelFormat.ONNX: [".onnx"],
        }

        # Search for model file
        for ext in extensions[model_format]:
            model_path = self.cache_dir / f"{model_name}{ext}"
            if model_path.exists():
                return model_path

        raise FileNotFoundError(
            f"Model {model_name} not found in {self.cache_dir} "
            f"with extensions {extensions[model_format]}"
        )

    def _load_tensorrt_model(self, model_path: Path) -> TensorRTModel:
        """Load TensorRT engine."""
        logger.info(f"Loading TensorRT engine from {model_path}")

        # Read serialized engine
        with open(model_path, 'rb') as f:
            engine_data = f.read()

        # Deserialize engine
        engine = self._trt_runtime.deserialize_cuda_engine(engine_data)

        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {model_path}")

        return TensorRTModel(engine)

    def _load_pytorch_model(self, model_path: Path) -> nn.Module:
        """Load PyTorch model."""
        logger.info(f"Loading PyTorch model from {model_path}")

        model = torch.load(model_path, map_location='cpu')

        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()

        model.eval()

        return model

    def _load_onnx_model(self, model_path: Path):
        """Load ONNX model."""
        logger.info(f"Loading ONNX model from {model_path}")

        import onnxruntime as ort

        # Create inference session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(model_path), providers=providers)

        return session

    def _estimate_model_memory(self, model: Any, model_format: ModelFormat) -> float:
        """
        Estimate model memory usage in MB.

        Args:
            model: Loaded model
            model_format: Model format

        Returns:
            Estimated memory in MB
        """
        if model_format == ModelFormat.TENSORRT:
            return model.engine.device_memory_size / (1024 ** 2)

        elif model_format == ModelFormat.PYTORCH:
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return (param_size + buffer_size) / (1024 ** 2)

        else:
            # Fallback: use file size
            return 100.0  # Default estimate

    def _evict_if_needed(self, required_mb: float) -> None:
        """
        Evict models from cache if needed to free memory.

        Uses LRU eviction strategy.

        Args:
            required_mb: Required memory in MB
        """
        current_size = sum(info.memory_mb for info in self._model_info.values())

        while current_size + required_mb > self.max_cache_size_mb and self._models:
            # Evict least recently used model
            lru_model_name = next(iter(self._models))
            lru_size = self._model_info[lru_model_name].memory_mb

            logger.info(
                f"Evicting LRU model {lru_model_name} ({lru_size:.1f}MB) "
                f"to free memory"
            )

            del self._models[lru_model_name]
            del self._model_info[lru_model_name]

            current_size -= lru_size

    def _warmup_model(self, model_name: str, num_iterations: int = 3) -> None:
        """
        Warmup model with dummy inputs.

        Args:
            model_name: Model to warmup
            num_iterations: Number of warmup iterations
        """
        logger.info(f"Warming up model {model_name}")

        try:
            model = self._models[model_name]
            info = self._model_info[model_name]

            # Create dummy inputs based on model format
            if info.format == ModelFormat.TENSORRT:
                dummy_inputs = self._create_tensorrt_dummy_inputs(model)
            else:
                logger.warning(f"Warmup not implemented for format {info.format}")
                return

            # Run warmup iterations
            for i in range(num_iterations):
                if info.format == ModelFormat.TENSORRT:
                    model.infer(dummy_inputs)

            info.warmup_completed = True
            logger.info(f"Warmup complete for {model_name}")

        except Exception as e:
            logger.warning(f"Warmup failed for {model_name}: {e}")

    def _create_tensorrt_dummy_inputs(self, model: TensorRTModel) -> Dict[str, np.ndarray]:
        """Create dummy inputs for TensorRT model warmup."""
        dummy_inputs = {}

        for name, input_info in model.inputs.items():
            shape = input_info["shape"]
            dtype = input_info["dtype"]

            # Handle dynamic shapes
            shape = tuple(1 if s == -1 else s for s in shape)

            dummy_inputs[name] = np.random.randn(*shape).astype(dtype)

        return dummy_inputs

    def get_model(self, model_name: str) -> Any:
        """
        Get a loaded model.

        Args:
            model_name: Model name

        Returns:
            Loaded model object

        Raises:
            KeyError: If model not loaded
        """
        with self._lock:
            if model_name not in self._models:
                raise KeyError(f"Model {model_name} not loaded")

            # Move to end (most recently used)
            self._models.move_to_end(model_name)

            return self._models[model_name]

    def get_model_info(self, model_name: str) -> ModelInfo:
        """
        Get model metadata.

        Args:
            model_name: Model name

        Returns:
            Model information

        Raises:
            KeyError: If model not loaded
        """
        with self._lock:
            if model_name not in self._model_info:
                raise KeyError(f"Model {model_name} not loaded")

            return self._model_info[model_name]

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        with self._lock:
            return model_name in self._models

    def list_loaded_models(self) -> list[str]:
        """Get list of loaded model names."""
        with self._lock:
            return list(self._models.keys())

    def unload_model(self, model_name: str) -> None:
        """
        Unload a model from cache.

        Args:
            model_name: Model to unload

        Raises:
            KeyError: If model not loaded
        """
        with self._lock:
            if model_name not in self._models:
                raise KeyError(f"Model {model_name} not loaded")

            logger.info(f"Unloading model {model_name}")

            del self._models[model_name]
            del self._model_info[model_name]

    def unload_all_models(self) -> None:
        """Unload all models from cache."""
        with self._lock:
            logger.info(f"Unloading all models ({len(self._models)} total)")

            self._models.clear()
            self._model_info.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_memory = sum(info.memory_mb for info in self._model_info.values())

            return {
                "num_models": len(self._models),
                "total_memory_mb": total_memory,
                "max_cache_size_mb": self.max_cache_size_mb,
                "cache_utilization": total_memory / self.max_cache_size_mb if self.max_cache_size_mb > 0 else 0,
                "models": [
                    {
                        "name": name,
                        "format": info.format.value,
                        "memory_mb": info.memory_mb,
                        "warmup_completed": info.warmup_completed,
                    }
                    for name, info in self._model_info.items()
                ]
            }


# Example usage
if __name__ == "__main__":
    loader = ModelLoader(cache_dir="/tmp/model_cache", max_cache_size_mb=2048)
    logger.info("ModelLoader ready for use")
