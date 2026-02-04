"""
vLLM Server Module

High-performance async server wrapper for vLLM inference engine with support
for streaming generation, batching, and comprehensive monitoring.
"""

import asyncio
import logging
import time
from typing import AsyncIterator, Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

try:
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.outputs import RequestOutput
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not installed. Install with: pip install vllm")

from .config import LLMConfig

logger = logging.getLogger(__name__)


class ServerState(Enum):
    """Server lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class GenerationRequest:
    """Request for text generation."""
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stream: bool = False
    request_id: Optional[str] = None


@dataclass
class GenerationResponse:
    """Response from text generation."""
    text: str
    request_id: str
    tokens_generated: int
    latency: float
    tokens_per_second: float
    finish_reason: str
    model: str


@dataclass
class ServerStats:
    """Server performance statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_generated: int = 0
    total_latency: float = 0.0
    average_latency: float = 0.0
    average_tokens_per_second: float = 0.0
    uptime: float = 0.0


class VLLMServer:
    """
    Async server wrapper for vLLM inference engine.

    This class provides a high-level interface to vLLM's AsyncLLMEngine with
    features like streaming generation, request tracking, performance monitoring,
    and graceful error handling.

    Attributes:
        config: LLM configuration
        engine: vLLM AsyncLLMEngine instance
        state: Current server state
        stats: Performance statistics

    Example:
        ```python
        config = LLMConfig(model="meta-llama/Llama-2-7b-hf")
        server = VLLMServer(config)

        await server.initialize()

        # Non-streaming generation
        response = await server.generate("Write a poem about AI")
        print(response.text)

        # Streaming generation
        async for chunk in server.generate_stream("Tell me a story"):
            print(chunk, end="", flush=True)

        await server.shutdown()
        ```
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize vLLM server.

        Args:
            config: LLM configuration instance
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM is not installed. Please install with: pip install vllm"
            )

        self.config = config
        self.engine: Optional[AsyncLLMEngine] = None
        self.state = ServerState.UNINITIALIZED
        self.stats = ServerStats()
        self._start_time: Optional[float] = None
        self._request_counter = 0
        self._lock = asyncio.Lock()

        logger.info(f"VLLMServer created with model: {config.model}")

    async def initialize(self) -> None:
        """
        Initialize the vLLM engine.

        This method loads the model and prepares the engine for inference.
        It may take several minutes for large models.

        Raises:
            RuntimeError: If initialization fails
        """
        if self.state == ServerState.READY:
            logger.warning("Server already initialized")
            return

        self.state = ServerState.INITIALIZING
        logger.info("Initializing vLLM engine...")

        try:
            # Convert config to engine args
            engine_args = AsyncEngineArgs(**self.config.to_engine_args())

            # Create async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)

            self.state = ServerState.READY
            self._start_time = time.time()

            logger.info(
                f"vLLM engine initialized successfully for model: {self.config.model}"
            )
            logger.info(
                f"Configuration: TP={self.config.tensor_parallel_size}, "
                f"max_seqs={self.config.max_num_seqs}"
            )

        except Exception as e:
            self.state = ServerState.ERROR
            logger.error(f"Failed to initialize vLLM engine: {e}", exc_info=True)
            raise RuntimeError(f"Engine initialization failed: {e}") from e

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        request_id: Optional[str] = None,
    ) -> GenerationResponse:
        """
        Generate text completion (non-streaming).

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop: Stop sequences
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            request_id: Optional request identifier

        Returns:
            GenerationResponse with completed text and metadata

        Raises:
            RuntimeError: If server is not ready
        """
        if self.state != ServerState.READY:
            raise RuntimeError(f"Server not ready. Current state: {self.state}")

        # Generate request ID
        if request_id is None:
            async with self._lock:
                self._request_counter += 1
                request_id = f"req_{self._request_counter}"

        logger.debug(f"Processing generation request {request_id}")

        start_time = time.time()

        try:
            # Create sampling parameters
            sampling_params = self._create_sampling_params(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

            # Generate
            results_generator = self.engine.generate(
                prompt,
                sampling_params,
                request_id=request_id
            )

            # Get final output
            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            if final_output is None:
                raise RuntimeError("No output generated")

            # Extract text and metadata
            output = final_output.outputs[0]
            text = output.text
            tokens_generated = len(output.token_ids)
            finish_reason = output.finish_reason

            latency = time.time() - start_time
            tokens_per_second = tokens_generated / latency if latency > 0 else 0

            # Update statistics
            async with self._lock:
                self.stats.total_requests += 1
                self.stats.successful_requests += 1
                self.stats.total_tokens_generated += tokens_generated
                self.stats.total_latency += latency
                self.stats.average_latency = (
                    self.stats.total_latency / self.stats.successful_requests
                )
                self.stats.average_tokens_per_second = (
                    self.stats.total_tokens_generated / self.stats.total_latency
                )

            response = GenerationResponse(
                text=text,
                request_id=request_id,
                tokens_generated=tokens_generated,
                latency=latency,
                tokens_per_second=tokens_per_second,
                finish_reason=finish_reason,
                model=self.config.model,
            )

            logger.debug(
                f"Request {request_id} completed: {tokens_generated} tokens "
                f"in {latency:.2f}s ({tokens_per_second:.1f} tok/s)"
            )

            return response

        except Exception as e:
            async with self._lock:
                self.stats.total_requests += 1
                self.stats.failed_requests += 1

            logger.error(f"Generation failed for request {request_id}: {e}", exc_info=True)
            raise

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Generate text completion with streaming output.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop: Stop sequences
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            request_id: Optional request identifier

        Yields:
            Generated text chunks as they become available

        Raises:
            RuntimeError: If server is not ready
        """
        if self.state != ServerState.READY:
            raise RuntimeError(f"Server not ready. Current state: {self.state}")

        # Generate request ID
        if request_id is None:
            async with self._lock:
                self._request_counter += 1
                request_id = f"req_{self._request_counter}"

        logger.debug(f"Processing streaming request {request_id}")

        start_time = time.time()
        tokens_generated = 0

        try:
            # Create sampling parameters
            sampling_params = self._create_sampling_params(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

            # Generate with streaming
            results_generator = self.engine.generate(
                prompt,
                sampling_params,
                request_id=request_id
            )

            previous_text = ""
            async for request_output in results_generator:
                output = request_output.outputs[0]
                current_text = output.text

                # Yield only the new text
                new_text = current_text[len(previous_text):]
                if new_text:
                    yield new_text

                previous_text = current_text
                tokens_generated = len(output.token_ids)

            # Update statistics
            latency = time.time() - start_time
            async with self._lock:
                self.stats.total_requests += 1
                self.stats.successful_requests += 1
                self.stats.total_tokens_generated += tokens_generated
                self.stats.total_latency += latency
                self.stats.average_latency = (
                    self.stats.total_latency / self.stats.successful_requests
                )
                self.stats.average_tokens_per_second = (
                    self.stats.total_tokens_generated / self.stats.total_latency
                )

            logger.debug(
                f"Streaming request {request_id} completed: {tokens_generated} tokens "
                f"in {latency:.2f}s"
            )

        except Exception as e:
            async with self._lock:
                self.stats.total_requests += 1
                self.stats.failed_requests += 1

            logger.error(f"Streaming generation failed for request {request_id}: {e}")
            raise

    def _create_sampling_params(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> SamplingParams:
        """Create vLLM sampling parameters with defaults from config."""
        return SamplingParams(
            max_tokens=max_tokens or self.config.default_max_tokens,
            temperature=temperature or self.config.default_temperature,
            top_p=top_p or self.config.default_top_p,
            top_k=top_k if top_k is not None else self.config.default_top_k,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

    async def get_stats(self) -> ServerStats:
        """
        Get current server statistics.

        Returns:
            ServerStats with current performance metrics
        """
        async with self._lock:
            stats = ServerStats(
                total_requests=self.stats.total_requests,
                successful_requests=self.stats.successful_requests,
                failed_requests=self.stats.failed_requests,
                total_tokens_generated=self.stats.total_tokens_generated,
                total_latency=self.stats.total_latency,
                average_latency=self.stats.average_latency,
                average_tokens_per_second=self.stats.average_tokens_per_second,
                uptime=time.time() - self._start_time if self._start_time else 0.0,
            )
        return stats

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Dictionary with health status and diagnostics
        """
        stats = await self.get_stats()

        return {
            "status": "healthy" if self.state == ServerState.READY else "unhealthy",
            "state": self.state.value,
            "model": self.config.model,
            "uptime": stats.uptime,
            "total_requests": stats.total_requests,
            "success_rate": (
                stats.successful_requests / stats.total_requests
                if stats.total_requests > 0 else 0.0
            ),
            "average_latency": stats.average_latency,
            "average_tokens_per_second": stats.average_tokens_per_second,
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        logger.info("Shutting down vLLM server...")
        self.state = ServerState.SHUTDOWN

        if self.engine is not None:
            # vLLM cleanup if needed
            pass

        logger.info("vLLM server shutdown complete")
