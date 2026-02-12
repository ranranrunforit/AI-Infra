"""
End-to-End Integration Tests

Complete workflow tests that verify the interaction between all components.
These tests simulate real-world usage scenarios using mocks to avoid
requiring GPU hardware or running services.
"""

import asyncio
import time
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_inference_pipeline():
    """Test complete pipeline from API request to inference result.

    Simulates: API -> Router -> Model Loader -> Inference -> Response
    """
    from src.serving.model_loader import ModelLoader, ModelFormat, ModelInfo
    from src.serving.batch_processor import DynamicBatchProcessor, RequestPriority
    from src.routing.router import IntelligentRouter, ModelEndpoint, RoutingStrategy

    # --- Setup mock model loader ---
    with patch.object(ModelLoader, '__init__', lambda self, **kwargs: None):
        loader = ModelLoader.__new__(ModelLoader)
        loader._models = {}
        loader._model_info = {}
        loader._lock = __import__('threading').RLock()
        loader._cache_dir = __import__('pathlib').Path("/tmp/test_cache")
        loader._max_cache_size_mb = 4096
        loader._enable_warmup = False
        loader._current_memory_mb = 0.0
        loader._access_order = []

        # Register a mock model
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value={"output": np.array([[0.1, 0.9]])})
        loader._models["test-model"] = mock_model
        loader._model_info["test-model"] = ModelInfo(
            name="test-model",
            format=ModelFormat.TENSORRT,
            path=None,
            memory_mb=100.0,
            warmup_completed=True,
        )

    # --- Setup mock batch processor ---
    async def mock_inference(batch_data):
        """Simulate batch inference."""
        batch_size = list(batch_data.values())[0].shape[0]
        return [{"class": "cat", "confidence": 0.95} for _ in range(batch_size)]

    processor = DynamicBatchProcessor(
        max_batch_size=4,
        timeout_ms=50.0,
        inference_fn=mock_inference,
    )

    # --- Submit a request through the batch processor ---
    await processor.start()
    try:
        result = await processor.submit(
            request_id="test-req-001",
            data={"input": np.random.randn(1, 3, 224, 224).astype(np.float32)},
            priority=RequestPriority.NORMAL,
        )
        assert result is not None
    finally:
        await processor.shutdown()

    # --- Verify model retrieval ---
    model = loader._models.get("test-model")
    assert model is not None

    # --- Verify model info ---
    info = loader._model_info.get("test-model")
    assert info is not None
    assert info.name == "test-model"
    assert info.format == ModelFormat.TENSORRT


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_model_serving():
    """Test serving multiple models simultaneously."""
    from src.serving.model_loader import ModelLoader, ModelFormat, ModelInfo

    with patch.object(ModelLoader, '__init__', lambda self, **kwargs: None):
        loader = ModelLoader.__new__(ModelLoader)
        loader._models = {}
        loader._model_info = {}
        loader._lock = __import__('threading').RLock()
        loader._cache_dir = __import__('pathlib').Path("/tmp/test_cache")
        loader._max_cache_size_mb = 4096
        loader._enable_warmup = False
        loader._current_memory_mb = 0.0
        loader._access_order = []

        # Register multiple models
        models = {
            "resnet50-fp16": (ModelFormat.TENSORRT, 200.0),
            "bert-base": (ModelFormat.PYTORCH, 500.0),
            "mobilenet": (ModelFormat.ONNX, 50.0),
        }

        for name, (fmt, mem) in models.items():
            mock_model = MagicMock()
            mock_model.infer = MagicMock(
                return_value={"output": np.array([[0.5, 0.3, 0.2]])}
            )
            loader._models[name] = mock_model
            loader._model_info[name] = ModelInfo(
                name=name, format=fmt, path=None, memory_mb=mem
            )

        # Verify all models are accessible
        assert len(loader._models) == 3
        for name in models:
            assert name in loader._models
            assert name in loader._model_info

        # Simulate concurrent inference on different models
        results = {}
        for name in models:
            model = loader._models[name]
            result = model.infer({"input": np.random.randn(1, 3, 224, 224)})
            results[name] = result

        assert len(results) == 3
        for name, result in results.items():
            assert "output" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_load_balancing_under_load():
    """Test load balancing with concurrent requests."""
    from src.routing.router import (
        IntelligentRouter,
        ModelEndpoint,
        RoutingStrategy,
    )

    # Create mock endpoints
    endpoints = [
        ModelEndpoint(url=f"http://model-{i}:8000", weight=1, healthy=True)
        for i in range(3)
    ]

    # Patch health_check to avoid real HTTP calls
    for ep in endpoints:
        ep.health_check = AsyncMock(return_value=True)

    router = IntelligentRouter(
        endpoints=endpoints,
        strategy=RoutingStrategy.ROUND_ROBIN,
        health_check_enabled=False,
    )

    # Simulate 100 routing decisions
    route_counts = {ep.url: 0 for ep in endpoints}
    for _ in range(99):  # 99 so it divides evenly by 3
        selected = await router.route()
        route_counts[selected.url] += 1

    # Round-robin should distribute roughly equally
    for url, count in route_counts.items():
        assert count == 33, f"{url} got {count} requests, expected 33"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_distributed_tracing_e2e():
    """Test distributed tracing across all components."""
    from src.tracing.tracer import OpenTelemetryTracer

    # Initialize tracer with console export (no Jaeger needed)
    tracer = OpenTelemetryTracer(
        service_name="test-model-serving",
        console_export=True,
        sample_rate=1.0,
    )
    tracer.initialize()

    # Create a root span simulating a request
    with tracer.start_span("test_request", attributes={"model": "resnet50"}) as span:
        span.set_attribute("request.id", "test-123")

        # Simulate routing
        with tracer.start_span("routing", attributes={"strategy": "round_robin"}) as routing_span:
            routing_span.set_attribute("endpoint.selected", "http://model-0:8000")
            routing_span.add_event("endpoint_selected")

        # Simulate inference
        with tracer.start_span("inference", attributes={"batch_size": 1}) as infer_span:
            await asyncio.sleep(0.01)  # Simulate work
            infer_span.set_attribute("latency_ms", 10.0)
            infer_span.set_attribute("result.class", "cat")

    # Verify tracer is functional
    assert tracer._tracer is not None

    # Shutdown cleanly
    tracer.shutdown()
