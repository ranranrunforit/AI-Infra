"""
Models Module - Common Model Definitions and Configurations

This module provides shared data structures and configurations used across
the model serving infrastructure. Includes model metadata, configuration
classes, and common model definitions.

Components:
    - ModelConfig: Model configuration and parameters
    - ModelMetadata: Model metadata and versioning information
    - InferenceRequest: Standardized inference request format
    - InferenceResponse: Standardized inference response format

Example:
    ```python
    from models import ModelConfig, ModelMetadata, InferenceRequest

    # Define model metadata
    metadata = ModelMetadata(
        name="llama-2-7b",
        version="1.0.0",
        framework="pytorch",
        task="text-generation"
    )

    # Configure model
    config = ModelConfig(
        model_path="/models/llama-2-7b",
        metadata=metadata,
        max_batch_size=32,
        device="cuda"
    )

    # Create inference request
    request = InferenceRequest(
        model=config.metadata.name,
        prompt="Once upon a time",
        max_tokens=100
    )
    ```
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import time


class ModelFramework(Enum):
    """Supported ML frameworks."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    VLLM = "vllm"
    TENSORRT = "tensorrt"


class ModelTask(Enum):
    """Model task types."""
    TEXT_GENERATION = "text-generation"
    TEXT_CLASSIFICATION = "text-classification"
    QUESTION_ANSWERING = "question-answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    EMBEDDING = "embedding"
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"


@dataclass
class ModelMetadata:
    """
    Model metadata and versioning information.

    Attributes:
        name: Model identifier
        version: Model version
        framework: ML framework (pytorch, tensorflow, etc.)
        task: Model task type
        description: Human-readable description
        author: Model author or organization
        license: Model license
        tags: Searchable tags
        parameters: Model parameter count
        context_length: Maximum context length for sequence models
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    name: str
    version: str
    framework: ModelFramework
    task: ModelTask
    description: str = ""
    author: str = ""
    license: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: Optional[int] = None
    context_length: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "framework": self.framework.value,
            "task": self.task.value,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "tags": self.tags,
            "parameters": self.parameters,
            "context_length": self.context_length,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            **self.custom_metadata,
        }


@dataclass
class ModelConfig:
    """
    Model configuration for serving.

    Attributes:
        model_path: Path to model files
        metadata: Model metadata
        max_batch_size: Maximum batch size for inference
        device: Device for inference (cuda, cpu)
        device_map: Device mapping for multi-GPU
        dtype: Model data type (float16, float32, etc.)
        quantization: Quantization method (4bit, 8bit, None)
        optimization_level: Optimization level (0-3)
        enable_batching: Enable dynamic batching
        enable_caching: Enable KV cache
        timeout: Inference timeout in seconds
    """

    model_path: str
    metadata: ModelMetadata
    max_batch_size: int = 32
    device: str = "cuda"
    device_map: Optional[Dict[str, Any]] = None
    dtype: str = "float16"
    quantization: Optional[str] = None
    optimization_level: int = 1
    enable_batching: bool = True
    enable_caching: bool = True
    timeout: float = 60.0
    extra_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_path": self.model_path,
            "metadata": self.metadata.to_dict(),
            "max_batch_size": self.max_batch_size,
            "device": self.device,
            "device_map": self.device_map,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "optimization_level": self.optimization_level,
            "enable_batching": self.enable_batching,
            "enable_caching": self.enable_caching,
            "timeout": self.timeout,
            **self.extra_config,
        }


@dataclass
class InferenceRequest:
    """
    Standardized inference request format.

    Attributes:
        model: Model identifier
        prompt: Input text or prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        stop: Stop sequences
        stream: Enable streaming response
        request_id: Optional request identifier
        metadata: Additional request metadata
    """

    model: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    stop: Optional[List[str]] = None
    stream: bool = False
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            "model": self.model,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop": self.stop,
            "stream": self.stream,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }


@dataclass
class InferenceResponse:
    """
    Standardized inference response format.

    Attributes:
        model: Model identifier
        text: Generated text
        request_id: Request identifier
        tokens_generated: Number of tokens generated
        finish_reason: Reason for generation completion
        latency: Total latency in seconds
        tokens_per_second: Generation throughput
        timestamp: Response timestamp
        metadata: Additional response metadata
    """

    model: str
    text: str
    request_id: str
    tokens_generated: int
    finish_reason: str
    latency: float
    tokens_per_second: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "model": self.model,
            "text": self.text,
            "request_id": self.request_id,
            "tokens_generated": self.tokens_generated,
            "finish_reason": self.finish_reason,
            "latency": latency,
            "tokens_per_second": self.tokens_per_second,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# Export all classes
__all__ = [
    "ModelFramework",
    "ModelTask",
    "ModelMetadata",
    "ModelConfig",
    "InferenceRequest",
    "InferenceResponse",
]
