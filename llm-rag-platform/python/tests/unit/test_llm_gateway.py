"""
Unit Tests for LLM Gateway
Tests Gemini and vLLM providers with mocked API calls
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.llm.gateway import LLMGateway, GeminiProvider, vLLMProvider, LLMBackend, LLMResponse


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response"""
    mock_resp = MagicMock()
    mock_resp.text = "Machine learning enables computers to learn from data."
    mock_resp.usage_metadata.prompt_token_count = 50
    mock_resp.usage_metadata.candidates_token_count = 15
    return mock_resp


@pytest.fixture
def mock_vllm_response_data():
    """Mock vLLM API response JSON"""
    return {
        "choices": [{"message": {"content": "RAG reduces hallucinations by grounding responses."}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        "model": "mistral-7b",
    }


# ── GeminiProvider Tests ──────────────────────────────────────────────────────

class TestGeminiProvider:
    def test_is_available_with_key(self):
        """Provider should be available when API key is set"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("google.generativeai.configure"), \
                 patch("google.generativeai.GenerativeModel"):
                provider = GeminiProvider(api_key="test-key")
        assert provider.api_key == "test-key"

    def test_is_not_available_without_key(self):
        """Provider should not be available without API key"""
        provider = GeminiProvider(api_key="")
        assert not provider.is_available()

    @pytest.mark.asyncio
    async def test_generate_success(self, mock_gemini_response):
        """Generate should return LLMResponse on success"""
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_model_cls:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = mock_gemini_response
            mock_model_cls.return_value = mock_model

            provider = GeminiProvider(api_key="test-key", model="gemini-2.0-flash")
            provider._client = mock_model

            result = await provider.generate(
                prompt="What is machine learning?",
                system_prompt="Be concise.",
            )

        assert isinstance(result, LLMResponse)
        assert "machine learning" in result.text.lower()
        assert result.model == "gemini-2.0-flash"
        assert result.backend == LLMBackend.GEMINI

    @pytest.mark.asyncio
    async def test_generate_raises_when_unavailable(self):
        """Generate should raise RuntimeError when not configured"""
        provider = GeminiProvider(api_key="")
        with pytest.raises(RuntimeError, match="not available"):
            await provider.generate(prompt="test")


# ── vLLMProvider Tests ────────────────────────────────────────────────────────

class TestvLLMProvider:
    def test_is_available(self):
        provider = vLLMProvider(endpoint="http://localhost:8000/v1")
        assert provider.is_available()

    def test_endpoint_from_env(self):
        with patch.dict(os.environ, {"VLLM_ENDPOINT": "http://vllm-server:8000/v1"}):
            provider = vLLMProvider()
        assert "vllm-server" in provider.endpoint

    @pytest.mark.asyncio
    async def test_generate_success(self, mock_vllm_response_data):
        """Generate should call vLLM API and return LLMResponse"""
        import httpx

        provider = vLLMProvider(endpoint="http://localhost:8000/v1", model="mistral-7b")

        mock_http_resp = MagicMock()
        mock_http_resp.json.return_value = mock_vllm_response_data
        mock_http_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_http_resp)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await provider.generate(prompt="What is RAG?")

        assert isinstance(result, LLMResponse)
        assert result.backend == LLMBackend.VLLM
        assert result.total_tokens == 120


# ── LLMGateway Tests ──────────────────────────────────────────────────────────

class TestLLMGateway:
    def test_gateway_registers_gemini_when_key_set(self):
        """Gateway should register Gemini provider when API key is set"""
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel"):
            gateway = LLMGateway(
                backend="gemini",
                gemini_api_key="test-key",
            )
        assert LLMBackend.GEMINI in gateway.providers

    def test_gateway_registers_vllm_when_endpoint_set(self):
        """Gateway should register vLLM provider when endpoint is set"""
        gateway = LLMGateway(
            backend="gemini",
            vllm_endpoint="http://localhost:8000/v1",
        )
        assert LLMBackend.VLLM in gateway.providers

    def test_route_by_sensitivity_routes_pii_to_vllm(self):
        """PII queries should route to vLLM for data privacy"""
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel"):
            gateway = LLMGateway(
                backend="gemini",
                gemini_api_key="test-key",
                vllm_endpoint="http://localhost:8000/v1",
            )

        routed = gateway.route_by_sensitivity(contains_pii=True)
        assert routed == LLMBackend.VLLM

    def test_route_by_sensitivity_defaults_to_gemini_without_vllm(self):
        """Without vLLM, PII queries should still route to Gemini"""
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel"):
            gateway = LLMGateway(backend="gemini", gemini_api_key="test-key")

        routed = gateway.route_by_sensitivity(contains_pii=True)
        assert routed == LLMBackend.GEMINI

    def test_get_provider_raises_when_no_providers(self):
        """Should raise RuntimeError when no providers configured"""
        gateway = LLMGateway.__new__(LLMGateway)
        gateway.providers = {}
        gateway.backend = LLMBackend.GEMINI

        with pytest.raises(RuntimeError, match="No LLM providers"):
            gateway.get_provider()

    def test_available_backends(self):
        """Should return list of configured backends"""
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel"):
            gateway = LLMGateway(
                backend="gemini",
                gemini_api_key="test-key",
                vllm_endpoint="http://localhost:8000/v1",
            )
        backends = gateway.available_backends
        assert "gemini" in backends
        assert "vllm" in backends
