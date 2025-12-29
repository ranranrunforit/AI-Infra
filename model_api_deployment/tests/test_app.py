"""
API Endpoint Tests

This module contains unit tests for the REST API endpoints.

Run tests with: pytest tests/test_app.py

Author: AI Infrastructure Curriculum
License: MIT
"""

import pytest
import io
from PIL import Image

import sys
import os
# Add src to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import app, init_model
from config import config

# =========================================================================
# Test Fixtures
# =========================================================================

@pytest.fixture(scope='module')
def initialize_model():
    """Initialize model once for all tests."""
    init_model()
    yield

@pytest.fixture
def client(initialize_model):
    """
    Create test client with model initialized.

    Returns:
        Test client for making requests
    """
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_image():
    """
    Create sample image for testing.

    Returns:
        BytesIO object containing image data
    """
    # Create a simple red image
    image = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr


@pytest.fixture
def large_image():
    """
    Create large image for size limit testing.

    Returns:
        BytesIO object containing large image
    """
    # Create a very large image that will definitely exceed 10MB
    # Create a large image (e.g., 5000x5000)
    image = Image.new('RGB', (1000, 1000), color='blue')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=100)
    img_byte_arr.seek(0)
    return img_byte_arr


# =========================================================================
# Health Check Tests
# =========================================================================

def test_health_endpoint_returns_200(client):
    """Test that /health endpoint returns 200 OK."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.is_json


def test_health_endpoint_returns_healthy_status(client):
    """Test that /health returns healthy status."""
    response = client.get('/health')
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True


def test_health_endpoint_includes_model_name(client):
    """Test that /health includes model name."""
    response = client.get('/health')
    data = response.get_json()
    assert 'model_name' in data
    assert data['model_name'] in ['resnet50', 'mobilenet_v2']


# =========================================================================
# Info Endpoint Tests
# =========================================================================

def test_info_endpoint_returns_200(client):
    """Test that /info endpoint returns 200 OK."""
    response = client.get('/info')
    assert response.status_code == 200


def test_info_endpoint_includes_model_info(client):
    """Test that /info includes model information."""
    response = client.get('/info')
    data = response.get_json()
    assert 'model' in data
    assert 'name' in data['model']
    assert 'framework' in data['model']


def test_info_endpoint_includes_api_version(client):
    """Test that /info includes API version."""
    response = client.get('/info')
    data = response.get_json()
    assert 'api' in data
    assert 'version' in data['api']


def test_info_endpoint_includes_limits(client):
    """Test that /info includes request limits."""
    response = client.get('/info')
    data = response.get_json()
    assert 'limits' in data
    assert 'max_file_size_mb' in data['limits']
    assert 'timeout_seconds' in data['limits']


# =========================================================================
# Prediction Endpoint Tests - Success Cases
# =========================================================================

def test_predict_endpoint_with_valid_image(client, sample_image):
    """Test prediction with valid image."""
    response = client.post(
        '/predict',
        data={'file': (sample_image, 'test.jpg')},
        content_type='multipart/form-data'
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert 'predictions' in data


def test_predict_returns_correct_number_of_predictions(client, sample_image):
    """Test that prediction returns correct number of results."""
    # Default top_k
    response = client.post(
        '/predict',
        data={'file': (sample_image, 'test.jpg')},
        content_type='multipart/form-data'
    )
    data = response.get_json()
    assert len(data['predictions']) == 5
    
    # Custom top_k - need to create new image since BytesIO was consumed
    image2 = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    image2.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    response = client.post(
        '/predict',
        data={
            'file': (img_bytes, 'test.jpg'),
            'top_k': '3'
        },
        content_type='multipart/form-data'
    )
    data = response.get_json()
    assert len(data['predictions']) == 3


def test_prediction_format(client, sample_image):
    """Test prediction response format."""
    response = client.post(
        '/predict',
        data={'file': (sample_image, 'test.jpg')},
        content_type='multipart/form-data'
    )
    data = response.get_json()
    predictions = data['predictions']
    
    for i, pred in enumerate(predictions, start=1):
        assert 'class' in pred
        assert 'confidence' in pred
        assert 'rank' in pred
        assert 0 <= pred['confidence'] <= 1
        assert pred['rank'] == i


def test_prediction_includes_latency(client, sample_image):
    """Test that response includes latency measurement."""
    response = client.post(
        '/predict',
        data={'file': (sample_image, 'test.jpg')},
        content_type='multipart/form-data'
    )
    data = response.get_json()
    assert 'latency_ms' in data
    assert data['latency_ms'] > 0


def test_prediction_includes_correlation_id(client, sample_image):
    """Test that response includes correlation ID."""
    response = client.post(
        '/predict',
        data={'file': (sample_image, 'test.jpg')},
        content_type='multipart/form-data'
    )
    data = response.get_json()
    assert 'correlation_id' in data
    assert data['correlation_id'].startswith('req-')


# =========================================================================
# Prediction Endpoint Tests - Error Cases
# =========================================================================

def test_predict_without_file_returns_400(client):
    """Test that request without file returns 400."""
    response = client.post('/predict', data={})
    assert response.status_code == 400
    data = response.get_json()
    assert data['success'] is False
    assert data['error']['code'] == 'MISSING_FILE'


def test_predict_with_empty_file_returns_400(client):
    """Test that empty file returns 400."""
    empty_file = io.BytesIO(b'')
    response = client.post(
        '/predict',
        data={'file': (empty_file, 'empty.jpg')},
        content_type='multipart/form-data'
    )
    assert response.status_code == 400


def test_predict_with_large_file_returns_413(client, large_image):
    """Test that file exceeding size limit returns 413."""
    # PATCH: Temporarily set the limit to 1 byte
    original_limit = config.MAX_FILE_SIZE
    config.MAX_FILE_SIZE = 1

    try:
        response = client.post(
            '/predict',
            data={'file': (large_image, 'large.jpg')},
            content_type='multipart/form-data'
        )
        assert response.status_code == 413
        data = response.get_json()
        assert data['error']['code'] == 'FILE_TOO_LARGE'
    finally:
        # Restore the limit so other tests don't fail
        config.MAX_FILE_SIZE = original_limit


def test_predict_with_invalid_image_returns_400(client):
    """Test that invalid image file returns 400."""
    invalid_file = io.BytesIO(b'This is not an image')
    response = client.post(
        '/predict',
        data={'file': (invalid_file, 'fake.jpg')},
        content_type='multipart/form-data'
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data['error']['code'] == 'INVALID_IMAGE_FORMAT'


def test_predict_with_invalid_top_k_returns_400(client):
    """Test that invalid top_k parameter returns 400."""
    test_cases = ['-1', '0', '100', 'abc']
    for invalid_top_k in test_cases:
        # Create fresh image for each test
        image = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = client.post(
            '/predict',
            data={
                'file': (img_bytes, 'test.jpg'),
                'top_k': invalid_top_k
            },
            content_type='multipart/form-data'
        )
        assert response.status_code == 400, f"Failed for top_k={invalid_top_k}"


# =========================================================================
# Edge Case Tests
# =========================================================================

def test_predict_with_grayscale_image(client):
    """Test prediction with grayscale image."""
    grayscale = Image.new('L', (224, 224), color=128)
    img_bytes = io.BytesIO()
    grayscale.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    response = client.post(
        '/predict',
        data={'file': (img_bytes, 'gray.jpg')},
        content_type='multipart/form-data'
    )
    assert response.status_code == 200


def test_predict_with_rgba_image(client):
    """Test prediction with RGBA image (with transparency)."""
    rgba = Image.new('RGBA', (224, 224), color=(255, 0, 0, 128))
    img_bytes = io.BytesIO()
    rgba.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    response = client.post(
        '/predict',
        data={'file': (img_bytes, 'rgba.png')},
        content_type='multipart/form-data'
    )
    assert response.status_code == 200


def test_predict_with_different_image_formats(client):
    """Test prediction with various image formats."""
    formats = [('JPEG', 'test.jpg'), ('PNG', 'test.png'), ('BMP', 'test.bmp')]
    
    for fmt, filename in formats:
        image = Image.new('RGB', (224, 224), color='green')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=fmt)
        img_bytes.seek(0)
        
        response = client.post(
            '/predict',
            data={'file': (img_bytes, filename)},
            content_type='multipart/form-data'
        )
        assert response.status_code == 200


# =========================================================================
# Concurrent Request Tests
# =========================================================================

def test_concurrent_predictions(client):  
    """Test handling of concurrent requests."""
    import threading
   
    results = []
    
    def make_request():
        # Create a FRESH BytesIO object for this specific thread/request
        # We use sample_image.getvalue() to get the raw bytes from the fixture
        img_copy = io.BytesIO(sample_image.getvalue())
        response = client.post(
            '/predict', 
            data={'file': (img_copy, 'test.jpg')},
            content_type='multipart/form-data'
        )
        results.append(response.status_code)
    
    threads = [threading.Thread(target=make_request) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert all(code == 200 for code in results)

# =========================================================================
# Performance Tests
# =========================================================================

def test_health_check_latency(client):
    """Test that health check responds quickly."""
    import time
    start = time.time()
    response = client.get('/health')
    latency_ms = (time.time() - start) * 1000
    assert latency_ms < 100
    assert response.status_code == 200


def test_prediction_latency(client, sample_image):
    """Test that prediction completes within timeout."""
    response = client.post(
        '/predict',
        data={'file': (sample_image, 'test.jpg')},
        content_type='multipart/form-data'
    )
    data = response.get_json()
    assert data['latency_ms'] < 5000  # 5 seconds for first prediction


# =========================================================================
# Error Handler Tests
# =========================================================================

def test_404_for_nonexistent_endpoint(client):
    """Test that nonexistent endpoints return 404."""
    response = client.get('/nonexistent')
    assert response.status_code == 404


def test_405_for_wrong_method(client):
    """Test that wrong HTTP method returns 405."""
    response = client.get('/predict')
    assert response.status_code == 405


# =========================================================================
# Run Tests
# =========================================================================

if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, '-v'])