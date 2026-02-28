"""
LLM Gateway for Project 303 - Enterprise LLM Platform with RAG
Supports multiple backends: Gemini Pro (primary), vLLM (self-hosted), OpenAI (fallback)

Original Design: AWS vLLM + GPT-4
GCP Adaptation: Gemini Pro as primary, vLLM on GCP VM optional, OpenAI optional
Google Colab: Works with Gemini Pro via google-generativeai (no GPU required)
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMBackend(str, Enum):
    """Supported LLM backend types"""
    GEMINI = "gemini"       # Google Gemini Pro (primary, GCP + Colab)
    VLLM = "vllm"           # Self-hosted vLLM (GPU VM)
    OPENAI = "openai"       # OpenAI GPT-4 (optional commercial)


@dataclass
class LLMResponse:
    """Standardized LLM response across all backends"""
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    backend: str = ""


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response from the LLM"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is configured and available"""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Provider (Google AI Studio / Vertex AI)
# ─────────────────────────────────────────────────────────────────────────────


class GeminiProvider(LLMProvider):
    """
    Google Gemini Pro provider using google-generativeai SDK.

    Works on:
    - GCP Cloud Run (set GOOGLE_API_KEY secret)
    - Google Colab (free tier: gemini-2.0-flash; Pro: gemini-1.5-pro, gemini-2.0-pro)
    - Local development
    """

    # Gemini models - check available models with:
    # curl "https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_KEY" | grep '"name"'
    SUPPORTED_MODELS = [
        "gemini-3-flash-preview",        # Latest Gemini 3 Flash
        "gemini-3-pro-preview",          # Latest Gemini 3 Pro
        "gemini-2.5-flash",             # Stable, fast
        "gemini-2.5-pro",               # Most capable stable
        "gemini-2.0-flash",             # May be deprecated for new users
        "gemini-1.5-pro",               # Previous generation
        "gemini-1.5-flash",             # Previous generation
    ]

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.model_name = model or os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL)
        self._client = None

        if self.api_key:
            self._init_client()

    def _init_client(self):
        """Initialize the Gemini client"""
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.model_name,
            )
            logger.info(f"Gemini provider initialized with model: {self.model_name}")
        except ImportError:
            logger.error(
                "google-generativeai not installed. "
                "Install it with: pip install google-generativeai"
            )
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key and self._client is not None)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> LLMResponse:
        """Generate response using Gemini Pro"""
        import time
        import google.generativeai as genai  # type: ignore

        if not self.is_available():
            raise RuntimeError(
                "Gemini provider not available. Check GOOGLE_API_KEY is set."
            )

        start_time = time.time()

        # Build messages - Gemini uses a different format than OpenAI
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=0.95,
        )

        # System instruction is set via model initialization; for runtime we
        # prepend to the user prompt if needed.
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        try:
            response = await asyncio.to_thread(
                self._client.generate_content,
                full_prompt,
                generation_config=generation_config,
            )

            latency_ms = (time.time() - start_time) * 1000
            text = response.text

            # Token usage (Gemini provides these)
            usage_metadata = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0
            completion_tokens = getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0

            logger.info(
                f"Gemini generation complete | model={self.model_name} "
                f"tokens={prompt_tokens}+{completion_tokens} latency={latency_ms:.0f}ms"
            )

            return LLMResponse(
                text=text,
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms,
                backend=LLMBackend.GEMINI,
            )

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise


# ─────────────────────────────────────────────────────────────────────────────
# vLLM Provider (Self-hosted, OpenAI-compatible API)
# ─────────────────────────────────────────────────────────────────────────────


class vLLMProvider(LLMProvider):
    """
    vLLM provider using OpenAI-compatible REST API.

    Deploy vLLM on:
    - GCP Compute Engine Spot VM with GPU (L4 ~$100-150/month)
    - Google Colab Pro+ with A100 (uses vLLM locally on the Colab node)
    - Local machine with GPU

    vLLM supports: Llama 3 70B, Mistral 7B, Mixtral, etc.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: str = "llama-3-70b",
        api_key: str = "EMPTY",  # vLLM doesn't require a key by default
    ):
        self.endpoint = endpoint or os.getenv(
            "VLLM_ENDPOINT", "http://localhost:8000/v1"
        )
        self.model_name = model or os.getenv("VLLM_MODEL", "llama-3-70b")
        self.api_key = api_key

    def is_available(self) -> bool:
        return bool(self.endpoint)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> LLMResponse:
        """Generate response using vLLM (OpenAI-compatible)"""
        import time
        import httpx

        start_time = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.95,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.endpoint}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()

            latency_ms = (time.time() - start_time) * 1000
            choice = data["choices"][0]
            usage = data.get("usage", {})

            logger.info(
                f"vLLM generation complete | model={self.model_name} "
                f"tokens={usage.get('total_tokens', 0)} latency={latency_ms:.0f}ms"
            )

            return LLMResponse(
                text=choice["message"]["content"],
                model=self.model_name,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                latency_ms=latency_ms,
                backend=LLMBackend.VLLM,
            )

        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise


# ─────────────────────────────────────────────────────────────────────────────
# LLM Gateway (Routes requests to appropriate backend)
# ─────────────────────────────────────────────────────────────────────────────


class LLMGateway:
    """
    Unified LLM Gateway - routes to appropriate provider based on config.

    Routing logic (matches original ARCHITECTURE.md design):
    - 'gemini' backend → Gemini Pro (primary for GCP/Colab)
    - 'vllm' backend   → Self-hosted vLLM (Llama 3 70B, Mistral 7B)
    - 'openai' backend → OpenAI GPT-4 (optional commercial)

    Set LLM_BACKEND env var to switch between backends.
    Falls back to available provider if primary is not configured.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        vllm_endpoint: Optional[str] = None,
        vllm_model: Optional[str] = None,
    ):
        self.backend = LLMBackend(
            backend or os.getenv("LLM_BACKEND", LLMBackend.GEMINI)
        )

        # Initialize all available providers
        self.providers: Dict[LLMBackend, LLMProvider] = {}

        # Gemini (always try to initialize)
        gemini = GeminiProvider(
            api_key=gemini_api_key,
            model=gemini_model or os.getenv("GEMINI_MODEL", GeminiProvider.DEFAULT_MODEL),
        )
        if gemini.is_available():
            self.providers[LLMBackend.GEMINI] = gemini
            logger.info("Gemini provider registered")

        # vLLM (only if endpoint configured)
        if vllm_endpoint or os.getenv("VLLM_ENDPOINT"):
            vllm = vLLMProvider(endpoint=vllm_endpoint, model=vllm_model)
            self.providers[LLMBackend.VLLM] = vllm
            logger.info(f"vLLM provider registered at {vllm.endpoint}")

        if not self.providers:
            logger.warning(
                "No LLM providers configured! Set GOOGLE_API_KEY for Gemini "
                "or VLLM_ENDPOINT for self-hosted vLLM."
            )

        logger.info(f"LLM Gateway initialized | primary backend: {self.backend}")

    def get_provider(self, backend: Optional[LLMBackend] = None) -> LLMProvider:
        """Get the appropriate provider, falling back if primary unavailable"""
        target = backend or self.backend

        if target in self.providers:
            return self.providers[target]

        # Fallback to any available provider
        for fallback_backend, provider in self.providers.items():
            logger.warning(
                f"Primary backend {target} not available, "
                f"falling back to {fallback_backend}"
            )
            return provider

        raise RuntimeError(
            "No LLM providers available. "
            "Set GOOGLE_API_KEY (Gemini) or VLLM_ENDPOINT (vLLM)."
        )

    def route_by_sensitivity(self, contains_pii: bool) -> LLMBackend:
        """
        Route based on data sensitivity (mirrors ARCHITECTURE.md routing logic).
        - PII detected → self-hosted (vLLM) for data privacy
        - Public data → Gemini (cost-efficient, no GPU)
        """
        if contains_pii and LLMBackend.VLLM in self.providers:
            return LLMBackend.VLLM
        return self.backend

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        backend: Optional[LLMBackend] = None,
        contains_pii: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response using the configured backend.

        Args:
            prompt: The user prompt / RAG-augmented query
            system_prompt: Optional system instructions
            temperature: Generation randomness (0.0 - 1.0)
            max_tokens: Maximum tokens in response
            backend: Override the default backend
            contains_pii: If True, routes to self-hosted vLLM for privacy
        """
        # Smart routing based on PII / explicit backend override
        if backend:
            target_backend = backend
        elif contains_pii:
            target_backend = self.route_by_sensitivity(contains_pii=True)
        else:
            target_backend = self.backend

        provider = self.get_provider(target_backend)
        return await provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @property
    def available_backends(self) -> List[str]:
        return [b.value for b in self.providers.keys()]


# ─────────────────────────────────────────────────────────────────────────────
# Example / Smoke Test
# ─────────────────────────────────────────────────────────────────────────────


async def main():
    """Smoke test - requires GOOGLE_API_KEY env var"""
    import os

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY to test Gemini provider")
        return

    gateway = LLMGateway(backend="gemini", gemini_api_key=api_key)
    print(f"Available backends: {gateway.available_backends}")

    response = await gateway.generate(
        prompt="What is retrieval-augmented generation (RAG)?",
        system_prompt="You are a helpful AI assistant. Be concise.",
        max_tokens=200,
    )
    print(f"\nResponse ({response.model}):\n{response.text}")
    print(f"Tokens: {response.total_tokens} | Latency: {response.latency_ms:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
