"""
Model configuration for LLM serving - RTX 5070 Fixed (12GB VRAM)

Memory-conservative configurations that actually work!
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
    def rtx_5070_tiny(cls) -> "ModelConfig":
        """
        RECOMMENDED FOR 12GB - TinyLlama (Ultra-safe)
        Memory usage: ~2-3GB, very fast, guaranteed to work
        """
        return cls(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dtype="float16",
            quantization=None,
            max_model_len=2048,
            gpu_memory_utilization=0.40,  # Very conservative
            tensor_parallel_size=1,
        )

    @classmethod
    def rtx_5070_phi3(cls) -> "ModelConfig":
        """
        RTX 5070 with Phi-3 Mini (Better quality, works on 12GB!)
        Memory usage: ~4-5GB model + ~1-2GB KV cache
        """
        return cls(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            dtype="float16",
            quantization=None,
            max_model_len=1024,  # REDUCED: Smaller context = less KV cache memory
            gpu_memory_utilization=0.40,  # REDUCED: More conservative
            tensor_parallel_size=1,
            trust_remote_code=True,
        )

    @classmethod
    def rtx_5070_mistral_awq(cls) -> "ModelConfig":
        """
        RTX 5070 with Mistral-7B AWQ (High quality, tight fit)
        Memory usage: ~5-6GB
        """
        return cls(
            model_name="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
            dtype="float16",
            quantization="awq",
            max_model_len=2048,  # Reduced for safety
            gpu_memory_utilization=0.50,  # Conservative
            tensor_parallel_size=1,
        )

    @classmethod
    def rtx_5070_llama2_awq(cls) -> "ModelConfig":
        """
        RTX 5070 with Llama-2-7B AWQ
        Memory usage: ~4-5GB
        """
        return cls(
            model_name="TheBloke/Llama-2-7B-Chat-AWQ",
            dtype="float16",
            quantization="awq",
            max_model_len=2048,
            gpu_memory_utilization=0.45,
            tensor_parallel_size=1,
        )

    # Legacy/Standard configs (may not work on 12GB)
    @classmethod
    def llama2_7b_chat(cls) -> "ModelConfig":
        """Llama 2 7B Chat - NOT RECOMMENDED for 12GB"""
        return cls(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            dtype="float16",
            max_model_len=4096,
            gpu_memory_utilization=0.90,
        )

    @classmethod
    def llama2_7b_chat_quantized(cls) -> "ModelConfig":
        """Llama 2 7B Chat AWQ"""
        return cls(
            model_name="TheBloke/Llama-2-7B-Chat-AWQ",
            dtype="float16",
            quantization="awq",
            max_model_len=4096,
            gpu_memory_utilization=0.70,
        )

    @classmethod
    def mistral_7b_instruct(cls) -> "ModelConfig":
        """Mistral 7B - NOT RECOMMENDED for 12GB"""
        return cls(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            dtype="float16",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
        )

    @classmethod
    def tiny_llama(cls) -> "ModelConfig":
        """TinyLlama standard config"""
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

    # Aliases
    @classmethod
    def rtx_5070(cls) -> "ModelConfig":
        """Default RTX 5070 config - uses Phi-3 Mini (good quality)"""
        return cls.rtx_5070_phi3()

    @classmethod
    def rtx_5070_optimized(cls) -> "ModelConfig":
        """Optimized RTX 5070 config - uses Phi-3 (better quality)"""
        return cls.rtx_5070_phi3()

    @classmethod
    def laptop_5070(cls) -> "ModelConfig":
        """Legacy alias"""
        return cls.rtx_5070_tiny()

    def get_memory_estimate_gb(self) -> float:
        """Estimate GPU memory required"""
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
        total_memory_gb = model_memory_gb * 1.5  # Add 50% overhead for safety

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


# Predefined configurations - RTX 5070 (12GB VRAM)
PREDEFINED_CONFIGS = {
    # RTX 5070 configs (in order of memory usage)
    "rtx-5070-tiny": ModelConfig.rtx_5070_tiny(),      # 2-3GB, safest
    "rtx-5070-phi3": ModelConfig.rtx_5070_phi3(),      # 5-7GB, FIXED for 12GB
    "rtx-5070-mistral": ModelConfig.rtx_5070_mistral_awq(),
    "rtx-5070-llama": ModelConfig.rtx_5070_llama2_awq(),
    
    # Default aliases
    "rtx-5070": ModelConfig.rtx_5070(),  # Points to phi3 now
    "laptop-5070": ModelConfig.rtx_5070_phi3(),
    
    # Standard configs (may not work on 12GB)
    "llama2-7b-chat": ModelConfig.llama2_7b_chat(),
    "llama2-7b-chat-quantized": ModelConfig.llama2_7b_chat_quantized(),
    "mistral-7b-instruct": ModelConfig.mistral_7b_instruct(),
    "tiny-llama": ModelConfig.tiny_llama(),
    "mock": ModelConfig.mock_model(),
}


def get_config(name: str) -> ModelConfig:
    """Get predefined configuration by name"""
    if name not in PREDEFINED_CONFIGS:
        raise ValueError(
            f"Unknown config: {name}. Available: {list(PREDEFINED_CONFIGS.keys())}"
        )
    return PREDEFINED_CONFIGS[name]