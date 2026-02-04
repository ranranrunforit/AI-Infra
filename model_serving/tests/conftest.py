"""
Pytest Configuration and Fixtures

Provides shared fixtures for testing the model serving infrastructure:
- Mock TensorRT components
- Mock PyTorch models
- FastAPI test clients
- Database and Redis fixtures
- GPU mocking utilities
- Temporary file management
- Async test support

This module ensures consistent test setup and teardown across all test files.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import application components
from serving.model_loader import ModelLoader
from serving.batch_processor import DynamicBatchProcessor
from routing.router import IntelligentRouter, ModelEndpoint, RoutingStrategy
from tensorrt.converter import TensorRTConverter, ConversionConfig, PrecisionMode


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Temporary File Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def model_cache_dir(temp_dir):
    """Create a temporary model cache directory."""
    cache_dir = temp_dir / "model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def tensorrt_engine_file(temp_dir):
    """Create a mock TensorRT engine file."""
    engine_path = temp_dir / "test_model.trt"
    # Write some dummy data to simulate engine file
    with open(engine_path, 'wb') as f:
        f.write(b'\x00' * 1024)  # 1KB dummy file
    return engine_path


@pytest.fixture
def onnx_model_file(temp_dir):
    """Create a mock ONNX model file."""
    onnx_path = temp_dir / "test_model.onnx"
    # Write dummy ONNX data
    with open(onnx_path, 'wb') as f:
        f.write(b'ONNX' + b'\x00' * 1020)  # 1KB with ONNX header
    return onnx_path


# ============================================================================
# Mock PyTorch Model Fixtures
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN model for testing."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 56 * 56, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def simple_pytorch_model():
    """Create a simple PyTorch model for testing."""
    model = SimpleCNN(num_classes=10)
    model.eval()
    return model


@pytest.fixture
def pytorch_model_file(temp_dir, simple_pytorch_model):
    """Save a PyTorch model to a temporary file."""
    model_path = temp_dir / "test_model.pt"
    torch.save(simple_pytorch_model, model_path)
    return model_path


@pytest.fixture
def sample_input_tensor():
    """Create a sample input tensor for model testing."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_input_numpy():
    """Create a sample numpy input for model testing."""
    return np.random.randn(1, 3, 224, 224).astype(np.float32)


# ============================================================================
# Mock TensorRT Fixtures
# ============================================================================

class MockTensorRTEngine:
    """Mock TensorRT engine for testing."""

    def __init__(self):
        self.num_bindings = 2
        self.device_memory_size = 100 * 1024 * 1024  # 100MB
        self.num_optimization_profiles = 1
        self._bindings = {
            0: {"name": "input", "shape": (1, 3, 224, 224), "is_input": True},
            1: {"name": "output", "shape": (1, 10), "is_input": False}
        }

    def get_binding_name(self, index):
        return self._bindings[index]["name"]

    def get_binding_shape(self, index):
        return self._bindings[index]["shape"]

    def get_binding_dtype(self, index):
        return 0  # TensorRT DataType.FLOAT

    def binding_is_input(self, index):
        return self._bindings[index]["is_input"]

    def create_execution_context(self):
        return MockTensorRTContext(self)

    def serialize(self):
        return b'\x00' * 1024  # Mock serialized data


class MockTensorRTContext:
    """Mock TensorRT execution context."""

    def __init__(self, engine):
        self.engine = engine

    def execute_async_v2(self, bindings, stream_handle):
        return True

    def get_binding_shape(self, index):
        return self.engine.get_binding_shape(index)


class MockTensorRTLogger:
    """Mock TensorRT logger."""

    def __init__(self, severity=0):
        self.severity = severity


class MockTensorRTBuilder:
    """Mock TensorRT builder."""

    def __init__(self, logger):
        self.logger = logger
        self.platform_has_fast_fp16 = True
        self.platform_has_fast_int8 = True

    def create_network(self, flags):
        return MockTensorRTNetwork()

    def create_builder_config(self):
        return MockTensorRTBuilderConfig()

    def build_serialized_network(self, network, config):
        return b'\x00' * 1024  # Mock serialized network

    def create_optimization_profile(self):
        return MockTensorRTOptimizationProfile()


class MockTensorRTNetwork:
    """Mock TensorRT network."""

    def __init__(self):
        self.num_inputs = 1
        self.num_outputs = 1
        self.num_layers = 10

    def get_input(self, index):
        mock_tensor = MagicMock()
        mock_tensor.name = "input"
        mock_tensor.shape = (1, 3, 224, 224)
        return mock_tensor


class MockTensorRTBuilderConfig:
    """Mock TensorRT builder config."""

    def __init__(self):
        self.flags = []
        self.int8_calibrator = None
        self.builder_optimization_level = 3
        self.profiling_verbosity = 0
        self._timing_cache = None

    def set_memory_pool_limit(self, pool_type, size):
        pass

    def set_flag(self, flag):
        self.flags.append(flag)

    def add_optimization_profile(self, profile):
        pass

    def create_timing_cache(self, data):
        self._timing_cache = data
        return data

    def set_timing_cache(self, cache, ignore_mismatch=False):
        self._timing_cache = cache

    def get_timing_cache(self):
        return self._timing_cache


class MockTensorRTOptimizationProfile:
    """Mock TensorRT optimization profile."""

    def set_shape(self, input_name, min_shape, opt_shape, max_shape):
        pass


class MockTensorRTRuntime:
    """Mock TensorRT runtime."""

    def __init__(self, logger):
        self.logger = logger

    def deserialize_cuda_engine(self, serialized_engine):
        return MockTensorRTEngine()


@pytest.fixture
def mock_tensorrt():
    """Mock TensorRT module and its components."""
    with patch('tensorrt.Logger', MockTensorRTLogger), \
         patch('tensorrt.Builder', MockTensorRTBuilder), \
         patch('tensorrt.Runtime', MockTensorRTRuntime), \
         patch('tensorrt.NetworkDefinitionCreationFlag'):
        yield


@pytest.fixture
def mock_tensorrt_engine():
    """Create a mock TensorRT engine."""
    return MockTensorRTEngine()


# ============================================================================
# Mock CUDA/GPU Fixtures
# ============================================================================

@pytest.fixture
def mock_cuda():
    """Mock PyCUDA for testing without GPU."""
    mock_cuda_module = MagicMock()
    mock_cuda_module.Stream = MagicMock
    mock_cuda_module.mem_alloc = MagicMock(return_value=MagicMock())
    mock_cuda_module.memcpy_htod_async = MagicMock()
    mock_cuda_module.memcpy_dtoh_async = MagicMock()

    with patch.dict('sys.modules', {'pycuda': mock_cuda_module, 'pycuda.driver': mock_cuda_module}):
        yield mock_cuda_module


@pytest.fixture
def mock_torch_cuda():
    """Mock torch.cuda functionality."""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=1), \
         patch('torch.cuda.get_device_name', return_value='Mock GPU'):
        yield


# ============================================================================
# TensorRT Converter Fixtures
# ============================================================================

@pytest.fixture
def tensorrt_config():
    """Create a TensorRT conversion configuration."""
    return ConversionConfig(
        precision=PrecisionMode.FP16,
        max_batch_size=16,
        max_workspace_size=1 << 30,
        enable_dynamic_shapes=True,
        enable_timing_cache=False,  # Disable for testing
        optimization_level=3
    )


@pytest.fixture
def tensorrt_converter(mock_tensorrt, tensorrt_config):
    """Create a TensorRT converter with mocked dependencies."""
    return TensorRTConverter(tensorrt_config)


# ============================================================================
# Model Loader Fixtures
# ============================================================================

@pytest.fixture
def model_loader(model_cache_dir, mock_cuda):
    """Create a ModelLoader instance for testing."""
    return ModelLoader(
        cache_dir=model_cache_dir,
        max_cache_size_mb=1024,
        enable_warmup=False  # Disable warmup for faster tests
    )


# ============================================================================
# Batch Processor Fixtures
# ============================================================================

@pytest.fixture
async def batch_processor():
    """Create a DynamicBatchProcessor for testing."""
    async def mock_inference(batch_data):
        batch_size = list(batch_data.values())[0].shape[0]
        return [{"result": i} for i in range(batch_size)]

    processor = DynamicBatchProcessor(
        max_batch_size=8,
        timeout_ms=10,
        max_queue_size=100,
        inference_fn=mock_inference
    )

    await processor.start()
    yield processor
    await processor.shutdown()


# ============================================================================
# Router Fixtures
# ============================================================================

@pytest.fixture
def mock_endpoints():
    """Create mock model endpoints for routing tests."""
    return [
        ModelEndpoint(url="http://gpu1:8000", weight=2),
        ModelEndpoint(url="http://gpu2:8000", weight=1),
        ModelEndpoint(url="http://gpu3:8000", weight=1),
    ]


@pytest.fixture
async def intelligent_router(mock_endpoints):
    """Create an IntelligentRouter for testing."""
    router = IntelligentRouter(
        endpoints=mock_endpoints,
        strategy=RoutingStrategy.ROUND_ROBIN,
        health_check_enabled=False  # Disable for testing
    )

    await router.start()
    yield router
    await router.stop()


# ============================================================================
# FastAPI Fixtures
# ============================================================================

@pytest.fixture
def fastapi_app():
    """Create a FastAPI test application."""
    from serving.server import app
    return app


@pytest.fixture
def test_client(fastapi_app):
    """Create a FastAPI test client."""
    return TestClient(fastapi_app)


@pytest.fixture
async def async_test_client(fastapi_app):
    """Create an async FastAPI test client."""
    async with AsyncClient(app=fastapi_app, base_url="http://test") as client:
        yield client


# ============================================================================
# OpenTelemetry Fixtures
# ============================================================================

@pytest.fixture
def mock_otel():
    """Mock OpenTelemetry components."""
    mock_span = MagicMock()
    mock_span.set_attribute = MagicMock()
    mock_span.set_attributes = MagicMock()
    mock_span.add_event = MagicMock()
    mock_span.set_status = MagicMock()
    mock_span.record_exception = MagicMock()
    mock_span.end = MagicMock()
    mock_span.is_recording = MagicMock(return_value=True)

    mock_tracer = MagicMock()
    mock_tracer.start_span = MagicMock(return_value=mock_span)

    with patch('opentelemetry.trace.get_tracer', return_value=mock_tracer):
        yield mock_tracer


# ============================================================================
# Database Fixtures (for future use)
# ============================================================================

@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock_client = MagicMock()
    mock_client.get = MagicMock(return_value=None)
    mock_client.set = MagicMock(return_value=True)
    mock_client.delete = MagicMock(return_value=1)
    mock_client.exists = MagicMock(return_value=False)
    return mock_client


@pytest.fixture
def mock_database():
    """Mock database connection."""
    mock_db = MagicMock()
    mock_db.execute = MagicMock()
    mock_db.fetchall = MagicMock(return_value=[])
    mock_db.commit = MagicMock()
    return mock_db


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    yield
    # Cleanup code runs after each test
    temp_files = [
        "/tmp/model_temp.onnx",
        "/tmp/timing_cache.bin",
    ]
    for filepath in temp_files:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables after each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


# ============================================================================
# Helper Functions
# ============================================================================

@pytest.fixture
def create_mock_request():
    """Factory fixture for creating mock HTTP requests."""
    def _create(method="POST", url="/", headers=None, json_data=None):
        mock_request = MagicMock()
        mock_request.method = method
        mock_request.url = url
        mock_request.headers = headers or {}
        mock_request.json = MagicMock(return_value=json_data or {})
        return mock_request
    return _create


@pytest.fixture
def assert_tensors_equal():
    """Helper function to assert tensor equality."""
    def _assert(tensor1, tensor2, rtol=1e-5, atol=1e-8):
        if isinstance(tensor1, torch.Tensor):
            tensor1 = tensor1.detach().cpu().numpy()
        if isinstance(tensor2, torch.Tensor):
            tensor2 = tensor2.detach().cpu().numpy()

        np.testing.assert_allclose(tensor1, tensor2, rtol=rtol, atol=atol)

    return _assert
