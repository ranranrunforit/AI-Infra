"""
Model configuration for LLM serving - RTX 5070 Optimized (12GB VRAM)

Defines model loading parameters, quantization settings,
and optimization configurations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for LLM model"""

    # Model identification
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer_name: Optional[str] = None  # Defaults to model_name

    # Model precision and quantization
    dtype: str = "float16"  # float16, bfloat16, float32
    quantization: Optional[str] = None  # None, "awq", "gptq", "squeezellm"

    # Model constraints
    max_model_len: int = 4096  # Maximum sequence length
    gpu_memory_utilization: float = 0.90  # GPU memory to use (0.0-1.0)

    # vLLM specific
    use_vllm: bool = True
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism

    # Safety and trust
    trust_remote_code: bool = False

    # Storage
    cache_dir: Optional[str] = None  # HuggingFace cache directory

    @classmethod
    def rtx_5070_optimized(cls) -> "ModelConfig":
        """
        NVIDIA RTX 5070 Optimized (12GB VRAM) - RECOMMENDED
        Uses Phi-3 Mini (3.8B) with FP16
        Memory usage: ~5-6GB, excellent performance
        """
        return cls(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            dtype="float16",
            quantization=None,
            max_model_len=4096,
            gpu_memory_utilization=0.65,  # Conservative for 12GB
            tensor_parallel_size=1,
            trust_remote_code=True,
        )

    @classmethod
    def rtx_5070_mistral_awq(cls) -> "ModelConfig":
        """
        RTX 5070 with Mistral-7B AWQ quantized
        Memory usage: ~6-7GB, good quality
        """
        return cls(
            model_name="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
            dtype="float16",
            quantization="awq",
            max_model_len=3072,  # Reduced context for safety
            gpu_memory_utilization=0.70,
            tensor_parallel_size=1,
        )

    @classmethod
    def rtx_5070_llama2_awq(cls) -> "ModelConfig":
        """
        RTX 5070 with Llama-2-7B AWQ quantized
        Memory usage: ~5-6GB
        """
        return cls(
            model_name="TheBloke/Llama-2-7B-Chat-AWQ",
            dtype="float16",
            quantization="awq",
            max_model_len=2048,  # Reduced context for 12GB
            gpu_memory_utilization=0.65,
            tensor_parallel_size=1,
        )

    @classmethod
    def rtx_5070_tinyllama(cls) -> "ModelConfig":
        """
        RTX 5070 with TinyLlama (for testing)
        Memory usage: ~2-3GB, very fast
        """
        return cls(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dtype="float16",
            quantization=None,
            max_model_len=2048,
            gpu_memory_utilization=0.50,
            tensor_parallel_size=1,
        )

    @classmethod
    def llama2_7b_chat(cls) -> "ModelConfig":
        """Llama 2 7B Chat configuration (NOT RECOMMENDED for 12GB)"""
        return cls(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            dtype="float16",
            max_model_len=4096,
            gpu_memory_utilization=0.90,
        )

    @classmethod
    def llama2_7b_chat_quantized(cls) -> "ModelConfig":
        """Llama 2 7B Chat with AWQ quantization"""
        return cls(
            model_name="TheBloke/Llama-2-7B-Chat-AWQ",
            dtype="float16",
            quantization="awq",
            max_model_len=4096,
            gpu_memory_utilization=0.70,
        )

    @classmethod
    def mistral_7b_instruct(cls) -> "ModelConfig":
        """Mistral 7B Instruct configuration (NOT RECOMMENDED for 12GB)"""
        return cls(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            dtype="float16",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
        )

    @classmethod
    def tiny_llama(cls) -> "ModelConfig":
        """TinyLlama for testing (1.1B parameters)"""
        return cls(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dtype="float16",
            max_model_len=2048,
            gpu_memory_utilization=0.50,
        )

    @classmethod
    def mock_model(cls) -> "ModelConfig":
        """Mock model for testing without GPU"""
        return cls(
            model_name="gpt2",
            dtype="float32",
            max_model_len=1024,
            gpu_memory_utilization=0.50,
            use_vllm=False,
        )

    # Legacy aliases for backward compatibility
    @classmethod
    def laptop_5070(cls) -> "ModelConfig":
        """Legacy: redirects to RTX 5070 config"""
        return cls.rtx_5070_optimized()

    def get_memory_estimate_gb(self) -> float:
        """
        Estimate GPU memory required (rough approximation)
        """
        model_lower = self.model_name.lower()

        if "70b" in model_lower:
            params_billions = 70
        elif "13b" in model_lower:
            params_billions = 13
        elif "7b" in model_lower:
            params_billions = 7
        elif "3.8b" in model_lower or "phi-3" in model_lower:
            params_billions = 3.8
        elif "3b" in model_lower:
            params_billions = 3
        elif "1.1b" in model_lower or "1b" in model_lower:
            params_billions = 1.1
        elif "gpt2" in model_lower:
            params_billions = 0.124
        else:
            params_billions = 7

        # Bytes per parameter
        if self.quantization in ["awq", "gptq", "squeezellm"]:
            bytes_per_param = 0.5  # 4-bit
        elif self.dtype == "float32":
            bytes_per_param = 4
        elif self.dtype in ["float16", "bfloat16"]:
            bytes_per_param = 2
        else:
            bytes_per_param = 2

        # Calculate memory
        model_memory_gb = params_billions * bytes_per_param
        total_memory_gb = model_memory_gb * 1.4  # Add 40% overhead for 12GB card

        return total_memory_gb

    def validate(self) -> None:
        """Validate configuration"""
        if self.dtype not in ["float16", "bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )

        if self.max_model_len < 256:
            raise ValueError(f"max_model_len too small: {self.max_model_len}")

        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}"
            )


# Predefined configurations
PREDEFINED_CONFIGS = {
    # RTX 5070 optimized configs (12GB VRAM)
    "rtx-5070": ModelConfig.rtx_5070_optimized(),  # RECOMMENDED
    "rtx-5070-mistral": ModelConfig.rtx_5070_mistral_awq(),
    "rtx-5070-llama": ModelConfig.rtx_5070_llama2_awq(),
    "rtx-5070-tiny": ModelConfig.rtx_5070_tinyllama(),
    
    # Standard configs (may not fit in 12GB)
    "llama2-7b-chat": ModelConfig.llama2_7b_chat(),
    "llama2-7b-chat-quantized": ModelConfig.llama2_7b_chat_quantized(),
    "mistral-7b-instruct": ModelConfig.mistral_7b_instruct(),
    "tiny-llama": ModelConfig.tiny_llama(),
    "mock": ModelConfig.mock_model(),
    
    # Legacy aliases
    "laptop-5070": ModelConfig.rtx_5070_optimized(),
}


def get_config(name: str) -> ModelConfig:
    """Get predefined configuration by name"""
    if name not in PREDEFINED_CONFIGS:
        raise ValueError(
            f"Unknown config: {name}. Available: {list(PREDEFINED_CONFIGS.keys())}"
        )
    return PREDEFINED_CONFIGS[name]