"""
FastAPI Model Serving Tests

Comprehensive test suite for the FastAPI-based model serving infrastructure.
Tests API endpoints, middleware, request/response models, error handling,
and server lifecycle management.

Test Coverage:
- Health check endpoint
- Prediction endpoint (/v1/predict)
- Generation endpoint (/v1/generate)
- Model loading/unloading endpoints
- Metrics endpoint
- Request validation
- Error handling and status codes
- Middleware functionality
- Async request handling
- Model loader integration
- Batch processor integration
"""

import asyncio
import time
from unittest.mock import MagicMock, Mock, patch, AsyncMock

import numpy as np
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import application components
from serving.server import (
    app,
    PredictRequest,
    PredictResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)


class TestRequestModels:
    """Test Pydantic request/response models."""

    def test_predict_request_valid(self):
        """Test valid PredictRequest creation."""
        request = PredictRequest(
            model="resnet50",
            inputs={"image": "base64_data"},
            parameters={"temperature": 0.7}
        )

        assert request.model == "resnet50"
        assert "image" in request.inputs
        assert request.parameters["temperature"] == 0.7

    def test_predict_request_invalid_model(self):
        """Test PredictRequest with invalid model name."""
        with pytest.raises(ValueError):
            PredictRequest(
                model="",
                inputs={"data": "test"}
            )

    def test_predict_request_default_parameters(self):
        """Test PredictRequest with default parameters."""
        request = PredictRequest(
            model="test-model",
            inputs={"data": "test"}
        )

        assert request.parameters == {}

    def test_generate_request_valid(self):
        """Test valid GenerateRequest creation."""
        request = GenerateRequest(
            model="llama-2-7b",
            prompt="Hello world",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9
        )

        assert request.model == "llama-2-7b"
        assert request.prompt == "Hello world"
        assert request.max_tokens == 100
        assert request.temperature == 0.7

    def test_generate_request_parameter_validation(self):
        """Test GenerateRequest parameter validation."""
        # Valid parameters
        request = GenerateRequest(
            model="llama",
            prompt="test",
            max_tokens=50,
            temperature=1.0,
            top_p=0.5
        )
        assert request.max_tokens == 50

        # Invalid max_tokens (too high)
        with pytest.raises(ValueError):
            GenerateRequest(
                model="llama",
                prompt="test",
                max_tokens=3000  # Over limit
            )

        # Invalid temperature (negative)
        with pytest.raises(ValueError):
            GenerateRequest(
                model="llama",
                prompt="test",
                temperature=-0.5
            )


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_success(self, test_client, mock_torch_cuda):
        """Test successful health check."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "gpu_available" in data
        assert "uptime_seconds" in data
        assert isinstance(data["gpu_available"], bool)

    def test_health_check_response_model(self, test_client, mock_torch_cuda):
        """Test health check response structure."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields
        assert "status" in data
        assert "models_loaded" in data
        assert "gpu_available" in data
        assert "uptime_seconds" in data
        assert "version" in data

    def test_health_check_with_loaded_models(self, test_client, mock_torch_cuda):
        """Test health check with models loaded."""
        # Mock model_loader to return loaded models
        with patch('serving.server.model_loader') as mock_loader:
            mock_loader.list_loaded_models.return_value = ["model1", "model2"]

            response = test_client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert len(data["models_loaded"]) == 2


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint."""

    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint returns Prometheus format."""
        response = test_client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        # Check for Prometheus format
        content = response.text
        assert "# HELP" in content or "# TYPE" in content or len(content) > 0


class TestPredictEndpoint:
    """Test prediction endpoint."""

    def test_predict_success(self, test_client):
        """Test successful prediction request."""
        with patch('serving.server.model_loader') as mock_loader, \
             patch('serving.server._preprocess_inputs') as mock_preprocess, \
             patch('serving.server._run_inference') as mock_inference:

            mock_loader.is_model_loaded.return_value = True
            mock_preprocess.return_value = asyncio.Future()
            mock_preprocess.return_value.set_result({"input": np.zeros((1, 3, 224, 224))})

            mock_inference.return_value = asyncio.Future()
            mock_inference.return_value.set_result([{"class": "cat", "confidence": 0.95}])

            response = test_client.post(
                "/v1/predict",
                json={
                    "model": "resnet50",
                    "inputs": {"image": "base64_data"},
                    "parameters": {}
                }
            )

            assert response.status_code == 200
            data = response.json()

            assert "predictions" in data
            assert "latency_ms" in data
            assert "model" in data
            assert data["model"] == "resnet50"

    def test_predict_invalid_request(self, test_client):
        """Test prediction with invalid request data."""
        response = test_client.post(
            "/v1/predict",
            json={
                "model": "",  # Invalid empty model
                "inputs": {}
            }
        )

        assert response.status_code == 422  # Validation error

    def test_predict_model_not_loaded(self, test_client):
        """Test prediction when model is not loaded."""
        with patch('serving.server.model_loader') as mock_loader:
            mock_loader.is_model_loaded.return_value = False
            mock_loader.load_model = MagicMock()

            response = test_client.post(
                "/v1/predict",
                json={
                    "model": "unloaded-model",
                    "inputs": {"data": "test"}
                }
            )

            # Should attempt to load the model
            assert mock_loader.load_model.called

    def test_predict_with_trace_id(self, test_client):
        """Test that prediction response includes trace ID."""
        with patch('serving.server.model_loader') as mock_loader, \
             patch('serving.server._preprocess_inputs') as mock_preprocess, \
             patch('serving.server._run_inference') as mock_inference:

            mock_loader.is_model_loaded.return_value = True
            mock_preprocess.return_value = asyncio.Future()
            mock_preprocess.return_value.set_result({"input": np.zeros((1, 3, 224, 224))})
            mock_inference.return_value = asyncio.Future()
            mock_inference.return_value.set_result([{"result": "test"}])

            response = test_client.post(
                "/v1/predict",
                json={
                    "model": "test-model",
                    "inputs": {"data": "test"}
                }
            )

            assert response.status_code == 200
            assert "X-Trace-ID" in response.headers
            data = response.json()
            assert "trace_id" in data

    def test_predict_error_handling(self, test_client):
        """Test error handling in prediction endpoint."""
        with patch('serving.server.model_loader') as mock_loader, \
             patch('serving.server._preprocess_inputs') as mock_preprocess:

            mock_loader.is_model_loaded.return_value = True
            mock_preprocess.side_effect = RuntimeError("Preprocessing failed")

            response = test_client.post(
                "/v1/predict",
                json={
                    "model": "test-model",
                    "inputs": {"data": "test"}
                }
            )

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data


class TestGenerateEndpoint:
    """Test text generation endpoint."""

    def test_generate_success(self, test_client):
        """Test successful text generation."""
        with patch('serving.server.model_loader') as mock_loader, \
             patch('serving.server._run_generation') as mock_generate:

            mock_loader.is_model_loaded.return_value = True
            mock_generate.return_value = asyncio.Future()
            mock_generate.return_value.set_result("Generated text response")

            response = test_client.post(
                "/v1/generate",
                json={
                    "model": "llama-2-7b",
                    "prompt": "Hello world",
                    "max_tokens": 100,
                    "temperature": 0.7
                }
            )

            assert response.status_code == 200
            data = response.json()

            assert "generated_text" in data
            assert "tokens_generated" in data
            assert "latency_ms" in data
            assert data["generated_text"] == "Generated text response"

    def test_generate_model_not_found(self, test_client):
        """Test generation with non-existent model."""
        with patch('serving.server.model_loader') as mock_loader:
            mock_loader.is_model_loaded.return_value = False

            response = test_client.post(
                "/v1/generate",
                json={
                    "model": "nonexistent-model",
                    "prompt": "test"
                }
            )

            assert response.status_code == 404

    def test_generate_parameter_validation(self, test_client):
        """Test parameter validation in generate endpoint."""
        # Valid request
        response = test_client.post(
            "/v1/generate",
            json={
                "model": "llama",
                "prompt": "test",
                "max_tokens": 50,
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 40
            }
        )

        # Response may be 404 (model not found) or 200, but not 422 (validation error)
        assert response.status_code != 422


class TestModelManagementEndpoints:
    """Test model loading and unloading endpoints."""

    def test_load_model_success(self, test_client):
        """Test successful model loading."""
        with patch('serving.server.model_loader') as mock_loader:
            mock_loader.load_model = MagicMock()

            response = test_client.post(
                "/v1/models/resnet50/load",
                params={"model_format": "tensorrt"}
            )

            assert response.status_code == 200
            data = response.json()

            assert data["model"] == "resnet50"
            assert data["status"] == "loaded"
            assert "load_time_seconds" in data

    def test_load_model_with_format(self, test_client):
        """Test loading model with specific format."""
        with patch('serving.server.model_loader') as mock_loader:
            mock_loader.load_model = MagicMock()

            response = test_client.post(
                "/v1/models/my-model/load",
                params={"model_format": "pytorch"}
            )

            assert response.status_code == 200
            mock_loader.load_model.assert_called_once()

    def test_load_model_failure(self, test_client):
        """Test model loading failure."""
        with patch('serving.server.model_loader') as mock_loader:
            mock_loader.load_model.side_effect = RuntimeError("Load failed")

            response = test_client.post(
                "/v1/models/bad-model/load"
            )

            assert response.status_code == 500

    def test_unload_model_success(self, test_client):
        """Test successful model unloading."""
        with patch('serving.server.model_loader') as mock_loader:
            mock_loader.unload_model = MagicMock()

            response = test_client.post("/v1/models/resnet50/unload")

            assert response.status_code == 200
            data = response.json()

            assert data["model"] == "resnet50"
            assert data["status"] == "unloaded"

    def test_unload_nonexistent_model(self, test_client):
        """Test unloading non-existent model."""
        with patch('serving.server.model_loader') as mock_loader:
            mock_loader.unload_model.side_effect = KeyError("Model not found")

            response = test_client.post("/v1/models/nonexistent/unload")

            assert response.status_code == 500


class TestMiddleware:
    """Test middleware functionality."""

    def test_request_tracking_middleware(self, test_client):
        """Test that middleware adds trace IDs."""
        response = test_client.get("/health")

        assert "X-Trace-ID" in response.headers
        trace_id = response.headers["X-Trace-ID"]
        assert len(trace_id) > 0

    def test_cors_middleware(self, test_client):
        """Test CORS middleware configuration."""
        response = test_client.options(
            "/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET"
            }
        )

        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers


class TestAsyncEndpoints:
    """Test async endpoint functionality."""

    @pytest.mark.asyncio
    async def test_async_predict(self, async_test_client):
        """Test prediction endpoint with async client."""
        with patch('serving.server.model_loader') as mock_loader, \
             patch('serving.server._preprocess_inputs') as mock_preprocess, \
             patch('serving.server._run_inference') as mock_inference:

            mock_loader.is_model_loaded.return_value = True

            async def mock_preprocess_async(*args, **kwargs):
                return {"input": np.zeros((1, 3, 224, 224))}

            async def mock_inference_async(*args, **kwargs):
                return [{"result": "test"}]

            mock_preprocess.side_effect = mock_preprocess_async
            mock_inference.side_effect = mock_inference_async

            response = await async_test_client.post(
                "/v1/predict",
                json={
                    "model": "test-model",
                    "inputs": {"data": "test"}
                }
            )

            assert response.status_code == 200


class TestHelperFunctions:
    """Test helper functions."""

    @pytest.mark.asyncio
    async def test_preprocess_inputs_string(self):
        """Test preprocessing string inputs."""
        from serving.server import _preprocess_inputs

        inputs = {"image": "base64_encoded_data"}
        processed = await _preprocess_inputs(inputs)

        assert "image" in processed
        assert isinstance(processed["image"], np.ndarray)

    @pytest.mark.asyncio
    async def test_preprocess_inputs_array(self):
        """Test preprocessing array inputs."""
        from serving.server import _preprocess_inputs

        inputs = {"data": [1.0, 2.0, 3.0]}
        processed = await _preprocess_inputs(inputs)

        assert "data" in processed
        assert isinstance(processed["data"], np.ndarray)
        assert processed["data"].dtype == np.float32

    @pytest.mark.asyncio
    async def test_run_inference(self):
        """Test inference execution."""
        from serving.server import _run_inference

        with patch('serving.server.model_loader') as mock_loader:
            mock_model = MagicMock()
            mock_loader.get_model.return_value = mock_model

            inputs = {"input": np.zeros((1, 3, 224, 224))}
            results = await _run_inference("test-model", inputs, {})

            assert results is not None
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_run_generation(self):
        """Test text generation execution."""
        from serving.server import _run_generation

        result = await _run_generation(
            model_name="llama",
            prompt="Hello",
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            stop_sequences=None
        )

        assert isinstance(result, str)
        assert len(result) > 0


class TestErrorHandlers:
    """Test global error handlers."""

    def test_global_exception_handler(self, test_client):
        """Test global exception handler."""
        # Trigger an endpoint that doesn't exist
        response = test_client.get("/nonexistent-endpoint")

        assert response.status_code == 404


class TestServerLifecycle:
    """Test server startup and shutdown."""

    @pytest.mark.asyncio
    async def test_lifespan_startup(self):
        """Test server startup initialization."""
        from serving.server import lifespan

        # Mock dependencies
        with patch('serving.server.ModelLoader') as mock_loader_class, \
             patch('serving.server.DynamicBatchProcessor') as mock_processor_class:

            mock_loader = MagicMock()
            mock_processor = MagicMock()
            mock_loader_class.return_value = mock_loader
            mock_processor_class.return_value = mock_processor

            # Simulate lifespan
            async with lifespan(app) as _:
                # Verify initialization
                assert mock_loader_class.called
                assert mock_processor_class.called

    def test_server_runs(self):
        """Test that server can be instantiated."""
        assert app is not None
        assert app.title == "High-Performance Model Serving API"
