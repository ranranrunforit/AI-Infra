"""
Pytest fixtures for testing.
"""

import os
import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

# Set test environment variables before importing app
os.environ["MODEL_DEVICE"] = "cpu"
os.environ["LOG_LEVEL"] = "ERROR"
os.environ["ENABLE_METRICS"] = "false"

from src.api import app
from src.model import ResNetClassifier, initialize_model, cleanup_model


@pytest.fixture(scope="session", autouse=True)
def initialize_test_model():
    """
    Session-scoped fixture to initialize the model once for all tests.
    This runs automatically before any tests.
    """
    # Initialize the global model instance
    initialize_model(device="cpu")
    yield
    # Cleanup after all tests
    cleanup_model()

@pytest.fixture
def test_client():
    """
    Fixture providing FastAPI test client.

    Returns:
        TestClient: FastAPI test client
    """
    # Use raise_server_exceptions=False to get proper error responses in tests
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def sample_image():
    """
    Fixture providing a sample RGB image.

    Returns:
        PIL.Image: Sample 224x224 RGB image
    """
    # Create a simple RGB image
    image = Image.new("RGB", (224, 224), color=(73, 109, 137))
    return image


@pytest.fixture
def sample_image_bytes(sample_image):
    """
    Fixture providing sample image as bytes.

    Args:
        sample_image: Sample PIL Image

    Returns:
        bytes: Image encoded as JPEG bytes
    """
    buffer = io.BytesIO()
    sample_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_grayscale_image():
    """
    Fixture providing a sample grayscale image.

    Returns:
        PIL.Image: Sample 224x224 grayscale image
    """
    return Image.new("L", (224, 224), color=128)


@pytest.fixture
def large_image():
    """
    Fixture providing a large image.

    Returns:
        PIL.Image: Large 2048x2048 RGB image
    """
    return Image.new("RGB", (2048, 2048), color=(200, 100, 50))


@pytest.fixture
def model_instance():
    """
    Fixture providing a fresh ResNet model instance.

    Returns:
        ResNetClassifier: Unloaded model instance
    """
    return ResNetClassifier(device="cpu")


@pytest.fixture
def loaded_model(model_instance):
    """
    Fixture providing a loaded ResNet model.

    Args:
        model_instance: Model instance fixture

    Returns:
        ResNetClassifier: Loaded model instance
    """
    model_instance.load()
    yield model_instance
    model_instance.unload()


@pytest.fixture
def mock_image_url():
    """
    Fixture providing a mock image URL.

    Returns:
        str: Mock image URL
    """
    return "https://example.com/test-image.jpg"
