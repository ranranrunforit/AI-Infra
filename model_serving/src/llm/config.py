"""
LLM Configuration Module

Provides configuration management for vLLM engine with hardware-aware optimization,
model-specific presets, and validation logic.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import torch

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """
    Configuration for vLLM inference engine.

    This dataclass encapsulates all parameters needed to initialize and configure
    a vLLM AsyncLLMEngine instance with optimal settings for production serving.

    Attributes:
        model: HuggingFace model identifier or local path
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pipeline_parallel_size: Number of GPUs for pipeline parallelism
        dtype: Model data type (float16, bfloat16, float32)
        max_num_seqs: Maximum number of sequences to process in parallel
        max_model_len: Maximum sequence length (context window)
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        swap_space: CPU swap space in GB for offloading
        enable_chunked_prefill: Enable chunked prefill for long sequences
        max_num_batched_tokens: Maximum tokens per batch
        trust_remote_code: Allow execution of remote code from HuggingFace
        quantization: Quantization method (awq, gptq, squeezellm, None)
        enforce_eager: Disable CUDA graphs for debugging
        disable_log_stats: Disable logging of statistics
        enable_prefix_caching: Enable KV cache prefix caching
    """

    # Model configuration
    model: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    dtype: str = "float16"

    # Batching and throughput
    max_num_seqs: int = 256
    max_model_len: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None

    # Memory management
    gpu_memory_utilization: float = 0.90
    swap_space: int = 4
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True

    # Model loading
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = "auto"
    revision: Optional[str] = None

    # Quantization
    quantization: Optional[str] = None

    # Performance tuning
    enforce_eager: bool = False
    disable_log_stats: bool = False

    # Generation defaults
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = -1

    # Additional engine arguments
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
        self._set_defaults()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")

        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be >= 1")

        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0.0, 1.0]")

        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")

        if self.dtype not in ["float16", "bfloat16", "float32", "auto"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.quantization and self.quantization not in ["awq", "gptq", "squeezellm"]:
            raise ValueError(f"Invalid quantization method: {self.quantization}")

        logger.info(f"Configuration validated for model: {self.model}")

    def _set_defaults(self) -> None:
        """Set intelligent defaults based on configuration."""
        # Set max_num_batched_tokens if not specified
        if self.max_num_batched_tokens is None:
            self.max_num_batched_tokens = self.max_num_seqs * 512

        # Warn about potential issues
        total_parallelism = self.tensor_parallel_size * self.pipeline_parallel_size
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if total_parallelism > available_gpus:
            logger.warning(
                f"Total parallelism ({total_parallelism}) exceeds available GPUs ({available_gpus})"
            )

    def to_engine_args(self) -> Dict[str, Any]:
        """
        Convert configuration to vLLM AsyncEngineArgs dictionary.

        Returns:
            Dictionary of arguments suitable for vLLM AsyncEngineArgs
        """
        args = {
            "model": self.model,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "dtype": self.dtype,
            "max_num_seqs": self.max_num_seqs,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "swap_space": self.swap_space,
            "trust_remote_code": self.trust_remote_code,
            "quantization": self.quantization,
            "enforce_eager": self.enforce_eager,
            "disable_log_stats": self.disable_log_stats,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "enable_prefix_caching": self.enable_prefix_caching,
        }

        # Add optional fields
        if self.download_dir:
            args["download_dir"] = self.download_dir
        if self.revision:
            args["revision"] = self.revision
        if self.load_format:
            args["load_format"] = self.load_format

        # Merge additional engine kwargs
        args.update(self.engine_kwargs)

        return args


# Model-specific preset configurations
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "llama-7b": {
        "max_model_len": 4096,
        "max_num_seqs": 256,
        "dtype": "float16",
        "tensor_parallel_size": 1,
    },
    "llama-13b": {
        "max_model_len": 4096,
        "max_num_seqs": 128,
        "dtype": "float16",
        "tensor_parallel_size": 2,
    },
    "llama-70b": {
        "max_model_len": 4096,
        "max_num_seqs": 64,
        "dtype": "float16",
        "tensor_parallel_size": 4,
    },
    "mistral-7b": {
        "max_model_len": 8192,
        "max_num_seqs": 256,
        "dtype": "float16",
        "tensor_parallel_size": 1,
    },
    "mixtral-8x7b": {
        "max_model_len": 32768,
        "max_num_seqs": 128,
        "dtype": "float16",
        "tensor_parallel_size": 2,
    },
}


def get_optimal_config(
    model: str,
    available_gpus: Optional[int] = None,
    memory_per_gpu_gb: Optional[int] = None,
    target_throughput: Optional[int] = None,
    **overrides
) -> LLMConfig:
    """
    Generate hardware-aware optimal configuration for a given model.

    This function analyzes available hardware and generates an optimized vLLM
    configuration that maximizes throughput while respecting resource constraints.

    Args:
        model: HuggingFace model identifier
        available_gpus: Number of available GPUs (auto-detected if None)
        memory_per_gpu_gb: GPU memory in GB (auto-detected if None)
        target_throughput: Target requests per second (optional)
        **overrides: Additional configuration overrides

    Returns:
        Optimized LLMConfig instance

    Example:
        ```python
        config = get_optimal_config(
            model="meta-llama/Llama-2-7b-hf",
            available_gpus=4,
            target_throughput=100
        )
        ```
    """
    # Auto-detect GPU configuration
    if available_gpus is None:
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if memory_per_gpu_gb is None and torch.cuda.is_available():
        memory_per_gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    logger.info(f"Generating optimal config for {model} with {available_gpus} GPUs")

    # Detect model family and size
    model_lower = model.lower()
    preset_key = None

    for key in MODEL_PRESETS:
        if key in model_lower:
            preset_key = key
            break

    # Start with preset or defaults
    config_dict = MODEL_PRESETS.get(preset_key, {}).copy() if preset_key else {}

    # Adjust tensor parallelism based on available GPUs
    if available_gpus and "tensor_parallel_size" in config_dict:
        config_dict["tensor_parallel_size"] = min(
            config_dict["tensor_parallel_size"],
            available_gpus
        )

    # Adjust batch size based on memory
    if memory_per_gpu_gb and memory_per_gpu_gb < 40:
        # Reduce batch size for smaller GPUs
        if "max_num_seqs" in config_dict:
            config_dict["max_num_seqs"] = max(
                config_dict["max_num_seqs"] // 2,
                32
            )

    # Adjust for target throughput
    if target_throughput:
        # Increase batch size for higher throughput targets
        if target_throughput > 100:
            config_dict["max_num_seqs"] = 512
            config_dict["gpu_memory_utilization"] = 0.95
        elif target_throughput > 50:
            config_dict["max_num_seqs"] = 256

    # Apply overrides
    config_dict.update(overrides)

    # Create config
    config = LLMConfig(model=model, **config_dict)

    logger.info(f"Generated config: TP={config.tensor_parallel_size}, "
                f"max_seqs={config.max_num_seqs}, max_len={config.max_model_len}")

    return config
