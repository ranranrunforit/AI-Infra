"""
LLM Configuration Module

This module defines configuration classes for LLM serving with:
- Model selection and paths
- Quantization settings
- GPU resource allocation
- Generation parameters
- Performance tuning options

Learning Objectives:
1. Understand LLM configuration best practices
2. Learn about quantization trade-offs
3. Configure GPU memory management
4. Set optimal generation parameters
5. Implement validation for configuration
"""

from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings
import yaml
import json
import torch

logger = logging.getLogger(__name__)

class QuantizationMethod(str, Enum):
    """
    Supported quantization methods for model compression.

    Quantization reduces model size and memory requirements while
    maintaining reasonable accuracy.

    Methods:
    - NONE: No quantization (FP16/FP32)
    - AWQ: Activation-aware Weight Quantization (recommended)
    - GPTQ: Post-training quantization
    - BITSANDBYTES: 8-bit/4-bit quantization
    - SQUEEZELLM: Efficient quantization for LLMs
    """
    NONE = "none"
    AWQ = "awq"
    GPTQ = "gptq"
    BITSANDBYTES = "bitsandbytes"
    SQUEEZELLM = "squeezellm"


class LLMConfig(BaseSettings):
    """
    Main configuration class for LLM serving.

    This class uses Pydantic for:
    - Type validation
    - Environment variable loading
    - Default value management
    - Configuration validation

    Attributes:
        model_name: Hugging Face model name or local path
        model_path: Optional local cache path for model weights
        quantization_method: Quantization technique to use
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to allocate
        max_model_length: Maximum sequence length (context window)
        max_num_seqs: Maximum sequences to batch together
    """

    # ========================================================================
    # Model Selection
    # ========================================================================

    model_name: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        description="Hugging Face model identifier or local path"
    )

    model_path: Optional[Path] = Field(
        default=None,
        description="Local path to cached model weights"
    )

    download_model_on_startup: bool = Field(
        default=True,
        description="Download model if not found locally"
    )

    # ========================================================================
    # Quantization Configuration
    # ========================================================================

    quantization_method: QuantizationMethod = Field(
        default=QuantizationMethod.NONE,
        description="Quantization method to reduce model size"
    )

    load_in_8bit: bool = Field(
        default=False,
        description="Load model in 8-bit precision (requires bitsandbytes)"
    )

    load_in_4bit: bool = Field(
        default=False,
        description="Load model in 4-bit precision (requires bitsandbytes)"
    )

    # ========================================================================
    # Performance Optimization
    # ========================================================================

    use_flash_attention: bool = Field(
        default=True,
        description="Enable Flash Attention 2 for faster inference"
    )

    enable_prefix_caching: bool = Field(
        default=True,
        description="Cache common prompt prefixes (e.g., system prompts)"
    )

    enable_chunked_prefill: bool = Field(
        default=False,
        description="Process long prompts in chunks to reduce latency"
    )

    # ========================================================================
    # GPU Configuration
    # ========================================================================

    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for tensor parallelism"
    )

    pipeline_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for pipeline parallelism"
    )

    gpu_memory_utilization: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Fraction of GPU memory to use (0.0-1.0)"
    )

    max_num_seqs: int = Field(
        default=256,
        ge=1,
        description="Maximum number of sequences in a batch"
    )

    swap_space: int = Field(
        default=4,
        description="CPU swap space in GB"
    )

    # ========================================================================
    # Model Context Configuration
    # ========================================================================

    max_model_length: int = Field(
        default=4096,
        ge=128,
        description="Maximum sequence length (context window)"
    )

    block_size: int = Field(
        default=16,
        description="KV cache block size"
    )

    # ========================================================================
    # Generation Defaults
    # ========================================================================

    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default sampling temperature"
    )

    default_top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Default nucleus sampling parameter"
    )

    default_top_k: int = Field(
        default=50,
        ge=0,
        description="Default top-k sampling parameter"
    )

    default_max_tokens: int = Field(
        default=512,
        ge=1,
        description="Default maximum tokens to generate"
    )

    # ========================================================================
    # Advanced Options
    # ========================================================================

    trust_remote_code: bool = Field(
        default=True,
        description="Trust remote code for custom models"
    )

    enforce_eager: bool = Field(
        default=False,
        description="Disable CUDA graph for debugging"
    )

    dtype: str = Field(
        default="auto",
        description="Model data type (auto, float16, bfloat16, float32)"
    )

    # ========================================================================
    # Pydantic Configuration
    # ========================================================================

    class Config:
        env_prefix = "LLM_"  # Load from LLM_MODEL_NAME, etc.
        case_sensitive = False
        use_enum_values = True

    # ========================================================================
    # Validators
    # ========================================================================

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: Optional[Path]) -> Optional[Path]:
        """
        Validate model path exists if provided.
        """
        if v is not None and not v.exists():
            # Only warn, as it might be created later or be a placeholder
            logger.warning(f"Model path does not exist: {v}")
        return v

    @model_validator(mode="after")
    def validate_quantization_settings(self):
        """
        Validate quantization configuration is consistent.
        """
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("Cannot use both 8-bit and 4-bit quantization")
        
        if self.quantization_method == QuantizationMethod.BITSANDBYTES:
            if not (self.load_in_8bit or self.load_in_4bit):
                logger.warning("BITSANDBYTES selected but neither load_in_8bit nor load_in_4bit is True. Defaulting to 4-bit.")
                self.load_in_4bit = True

        return self

    @model_validator(mode="after")
    def validate_gpu_settings(self):
        """
        Validate GPU configuration is feasible.
        """
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            total_requested = self.tensor_parallel_size * self.pipeline_parallel_size
            if total_requested > available_gpus:
                logger.warning(f"Requested {total_requested} GPUs but only {available_gpus} available. Falling back to {available_gpus}.")
                self.tensor_parallel_size = available_gpus
                self.pipeline_parallel_size = 1
        return self

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def get_vllm_engine_args(self) -> dict:
        """
        Convert config to vLLM AsyncEngineArgs parameters.
        """
        quant = None
        if self.quantization_method != QuantizationMethod.NONE:
            quant = self.quantization_method
            if self.quantization_method == QuantizationMethod.BITSANDBYTES:
                 # vLLM handles bitsandbytes differently, typically via `load_format` or specific quantization args
                 # For simplicity here we pass it through, but vLLM mostly supports 'awq', 'gptq', 'squeezellm'
                 # 'bitsandbytes' might need 'load_format="bitsandbytes"' in newer versions
                 pass

        engine_args = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "quantization": quant,
            "max_model_len": self.max_model_length,
            "max_num_seqs": self.max_num_seqs,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            "enforce_eager": self.enforce_eager,
            "swap_space": self.swap_space,
            "block_size": self.block_size,
            "enable_prefix_caching": self.enable_prefix_caching,
            # "enable_chunked_prefill": self.enable_chunked_prefill, # Depends on vLLM version
        }
        
        if self.model_path:
             engine_args["download_dir"] = str(self.model_path)

        return engine_args

    def estimate_memory_usage(self) -> float:
        """
        Estimate GPU memory usage for this configuration in GB.
        Rough estimation.
        """
        # Base model size (billions of params)
        # Use a heuristic based on model name if possible, else default to 7B
        params_billion = 7.0
        if "70b" in self.model_name.lower():
            params_billion = 70.0
        elif "13b" in self.model_name.lower():
            params_billion = 13.0
        elif "mistral" in self.model_name.lower() or "7b" in self.model_name.lower():
            params_billion = 7.0
        
        bytes_per_param = 2 # FP16
        if self.quantization_method == QuantizationMethod.AWQ or self.quantization_method == QuantizationMethod.GPTQ:
            bytes_per_param = 0.5 # 4-bit roughly
        elif self.load_in_8bit:
            bytes_per_param = 1
        elif self.load_in_4bit:
            bytes_per_param = 0.5
        
        model_size_gb = params_billion * bytes_per_param
        
        # KV Cache estimation (very rough)
        # 2 * 2 * n_layers * n_heads * head_dim * seq_len * batch_size * element_size
        # Assuming 7B model: 32 layers, 32 heads, 128 dim
        kv_cache_gb = (2 * 32 * 32 * 128 * self.max_model_length * self.max_num_seqs * 2) / (1024**3)
        
        # Overhead (activations, library, etc) ~20%
        return (model_size_gb + kv_cache_gb) * 1.2

    def get_model_info(self) -> dict:
        """
        Get human-readable configuration summary.
        """
        return {
            "model": self.model_name,
            "quantization": self.quantization_method,
            "gpu_utilization": self.gpu_memory_utilization,
            "context_window": self.max_model_length,
            "max_batch_size": self.max_num_seqs,
            "estimated_memory_gb": f"{self.estimate_memory_usage():.2f} GB"
        }


class ChatConfig(LLMConfig):
    """
    Extended configuration for chat-specific LLM serving.
    """

    system_prompt: Optional[str] = Field(
        default=None,
        description="Default system prompt for chat"
    )

    chat_template_name: Optional[str] = Field(
        default=None,
        description="Chat template to use (chatml, llama2, etc.)"
    )

    stop_sequences: List[str] = Field(
        default_factory=list,
        description="Stop sequences for chat generation"
    )


# ============================================================================
# Configuration Presets
# ============================================================================

def get_config_preset(preset_name: str) -> LLMConfig:
    """
    Get predefined configuration presets for common scenarios.
    
    Current preset for 'laptop-5070' is tuned for NVIDIA RTX 5070 (approx 12-16GB VRAM).
    """
    if preset_name == "laptop-5070":
        return LLMConfig(
            model_name="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
            quantization_method=QuantizationMethod.AWQ,
            gpu_memory_utilization=0.9, # Maximize usage of the single GPU
            max_model_length=4096, # Reasonable context
            max_num_seqs=16, # Moderate batch size for desktop use
            tensor_parallel_size=1,
            dtype="float16"
        )
    elif preset_name == "development":
        return LLMConfig(
            quantization_method=QuantizationMethod.AWQ, # Use AWQ for speed/memory balance
            gpu_memory_utilization=0.7,
            max_num_seqs=4
        )
    elif preset_name == "production":
         return LLMConfig(
            gpu_memory_utilization=0.95,
            max_num_seqs=64,
            use_flash_attention=True
        )
    else:
        # Default return
        return LLMConfig()


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config_from_file(config_path: Path) -> LLMConfig:
    """
    Load configuration from YAML or JSON file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            data = json.load(f)
        else:
             raise ValueError("Unsupported config file format. Use .yaml or .json")
    
    return LLMConfig(**data)

