"""
LLM Module - vLLM Integration for High-Performance Model Serving

This module provides components for integrating vLLM engine with the model serving
infrastructure. It includes server wrappers, configuration management, and utilities
for efficient LLM inference.

Components:
    - VLLMServer: Async server wrapper for vLLM engine
    - LLMConfig: Configuration dataclass for vLLM parameters
    - get_optimal_config: Hardware-aware configuration helper

Example:
    ```python
    from llm import VLLMServer, LLMConfig

    config = LLMConfig(
        model="meta-llama/Llama-2-7b-hf",
        tensor_parallel_size=2,
        max_num_seqs=256
    )

    server = VLLMServer(config)
    await server.initialize()

    async for token in server.generate("Hello, world!", stream=True):
        print(token, end="", flush=True)
    ```
"""

from .config import LLMConfig, get_optimal_config
from .vllm_server import VLLMServer

__all__ = [
    "LLMConfig",
    "VLLMServer",
    "get_optimal_config",
]

__version__ = "1.0.0"
