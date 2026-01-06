"""
Production ML System - Integrated Application
==============================================

This module integrates all components from Projects 1-4 into a unified
production-ready ML serving application.

Components Integrated:
- Project 1: Model serving API with Flask/FastAPI
- Project 2: Kubernetes-ready configuration
- Project 3: MLflow integration for model loading
- Project 4: Prometheus metrics and structured logging

Author: AI Infrastructure Curriculum
Version: 1.0
"""

import os
import logging
import time
from typing import Dict, Any, Optional
from functools import wraps

# TODO: Import required libraries
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import mlflow
import torch  # or tensorflow
from PIL import Image
import io

import json
import torch
from torchvision import transforms

# ============================================================================
# CONFIGURATION
# ============================================================================


# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# TODO: Load configuration from environment variables
# Example:
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-server:5001')
MODEL_NAME = os.getenv('MODEL_NAME', 'image-classifier')
MODEL_VERSION = os.getenv('MODEL_VERSION', 'latest')
API_KEYS = os.getenv('API_KEYS', '').split(',')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# TODO: Set up structured logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PROMETHEUS METRICS INSTRUMENTATION
# ============================================================================

# TODO: Define Prometheus metrics
# Examples:
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_latency = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

prediction_count = Counter(
    'model_predictions_total',
    'Total model predictions',
    ['model_name', 'model_version', 'status']
)

prediction_latency = Histogram(
    'model_prediction_duration_seconds',
    'Model prediction latency',
    ['model_name', 'model_version']
)

model_version_info = Gauge(
    'model_version_info',
    'Current model version',
    ['model_name', 'version']
)


# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """
    Manages ML model loading, versioning, and inference.

    Responsibilities:
    - Load models from MLflow Model Registry
    - Cache models in memory for fast inference
    - Handle model versioning and updates
    - Provide thread-safe model access
    """

    def __init__(self, mlflow_uri: str, model_name: str, model_version: str = 'latest'):
        """
        Initialize the ModelManager.

        Args:
            mlflow_uri: MLflow tracking server URI
            model_name: Name of the model in MLflow registry
            model_version: Version or stage (e.g., 'latest', 'Production', '12')
        """
        # TODO: Initialize MLflow client
        self.mlflow_uri = mlflow_uri
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.model_metadata = {}

        # TODO: Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_uri)

        # TODO: Load model on initialization
        self.load_model()
        

    def load_model(self) -> None:
        """
        Load model from MLflow Model Registry.

        Steps:
        1. Connect to MLflow tracking server
        2. Query model registry for specified model version
        3. Download model artifacts
        4. Load model into memory
        5. Extract and store metadata (version, run_id, etc.)
        6. Update Prometheus gauge with model version
        """
        # TODO: Implement model loading logic
        # Example:
        try:
            logger.info(f"Loading model {self.model_name} version {self.model_version}")
            
            client = mlflow.tracking.MlflowClient()
            
            if self.model_version == 'latest':
                # Modern approach: Get all versions and sort by version number
                all_versions = client.search_model_versions(f"name='{self.model_name}'")
                
                if not all_versions:
                    raise ValueError(f"No model found for {self.model_name}")
                
                # Filter for versions with 'Production' alias if available
                production_versions = [v for v in all_versions if 'Production' in (v.aliases or [])]
                
                if production_versions:
                    # Use the latest production version
                    model_version_obj = max(production_versions, key=lambda v: int(v.version))
                else:
                    # Fallback: use the highest version number
                    model_version_obj = max(all_versions, key=lambda v: int(v.version))
                
                model_version = model_version_obj.version
            else:
                model_version = self.model_version
            
            # Load model
            model_uri = f"models:/{self.model_name}/{model_version}"
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model.eval()  # Set to evaluation mode
            
            # Get detailed metadata
            version_details = client.get_model_version(self.model_name, model_version)
            
            # Store metadata
            self.model_metadata = {
                'name': self.model_name,
                'version': model_version,
                'uri': model_uri,
                'run_id': version_details.run_id,
                'status': version_details.status,
                'aliases': version_details.aliases or []
            }
            
            # Update Prometheus metric
            model_version_info.labels(
                model_name=self.model_name,
                version=model_version
            ).set(1)
            
            logger.info(f"Model loaded successfully: {model_uri}")
            logger.info(f"Model metadata: {self.model_metadata}")
            
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            logger.warning("Using dummy model for demonstration")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a simple dummy model for testing"""
        class DummyModel:
            def __call__(self, x):
                # Return random predictions for demonstration
                import torch
                batch_size = x.shape[0]
                return torch.rand(batch_size, 1000)
            
            def eval(self):
                pass
        
        self.model = DummyModel()
        self.model_metadata = {
            'name': self.model_name,
            'version': 'dummy-v1.0',
            'uri': 'local://dummy-model'
        }
        
        # Update Prometheus metric
        model_version_info.labels(
            model_name=self.model_name,
            version='dummy-v1.0'
        ).set(1)
        

    def predict(self, input_data: Any) -> Dict[str, Any]:
        """
        Run model inference on input data.

        Args:
            input_data: Preprocessed input ready for model inference

        Returns:
            Dictionary containing predictions and metadata
        """
        # TODO: Implement prediction logic
        # Example:
        try:
            start_time = time.time()
        
            # Run inference
            with torch.no_grad():
                predictions = self.model(input_data)
        
            # Process predictions - convert to probabilities and get top-k
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=5, dim=1)
            
            # Convert to Python lists for JSON serialization
            results = []
            for i in range(input_data.shape[0]):  # Handle batch
                results.append({
                    'top_classes': top_indices[i].tolist(),
                    'top_probabilities': top_probs[i].tolist()
                })
        
            # Calculate latency
            latency = time.time() - start_time
        
            # Record metrics
            prediction_latency.labels(
                model_name=self.model_name,
                model_version=self.model_metadata.get('version', 'unknown')
            ).observe(latency)
        
            prediction_count.labels(
                model_name=self.model_name,
                model_version=self.model_metadata.get('version', 'unknown'),
                status='success'
            ).inc()
        
            return {
                'predictions': results,
                'model_version': self.model_metadata.get('version'),
                'model_name': self.model_name,
                'latency_ms': round(latency * 1000, 2)
            }
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            prediction_count.labels(
                model_name=self.model_name,
                model_version=self.model_metadata.get('version', 'unknown'),
                status='error'
            ).inc()
            raise
        

    def get_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.

        Returns:
            Dictionary with model name, version, and other metadata
        """
        # TODO: Return model metadata
        return {
            'model_name': self.model_name,
            'model_version': self.model_metadata.get('version', 'unknown'),
            'model_uri': self.model_metadata.get('uri', 'unknown'),
            'status': 'loaded' if self.model is not None else 'not_loaded'
        }
        


# ============================================================================
# AUTHENTICATION & AUTHORIZATION
# ============================================================================

def require_api_key(f):
    """
    Decorator to require API key authentication for endpoints.

    Checks for X-API-Key header and validates against allowed keys.
    Returns 401 if missing, 403 if invalid.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # TODO: Implement API key validation
        # Example:
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            logger.warning("API key missing in request")
            return jsonify({'error': 'API key required'}), 401
        
        if api_key not in API_KEYS:
            logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
            return jsonify({'error': 'Invalid API key'}), 403
        
        return f(*args, **kwargs)
        

    return decorated_function


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def validate_image_upload(file) -> bool:
    """
    Validate uploaded image file.

    Checks:
    - File is not None
    - File size < 10MB
    - File type is allowed (JPEG, PNG)
    - File is a valid image (can be opened)

    Args:
        file: Uploaded file object

    Returns:
        True if valid, raises ValueError otherwise
    """
    # TODO: Implement image validation
    # Example:
    if not file:
        raise ValueError("No file provided")
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    max_size = 10 * 1024 * 1024  # 10MB
    if file_size > max_size:
        raise ValueError(f"File too large: {file_size} bytes (max {max_size})")
    
    # Check file type (MIME type)
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
    # Use python-magic or check file extension
    
    # Verify it's a valid image
    try:
        img = Image.open(file)
        img.verify()
        file.seek(0)  # Reset file pointer after verify
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")
    
    return True
    


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_image(file) -> Any:
    """
    Preprocess image for model inference.

    Steps:
    1. Load image from file
    2. Resize to model input size (e.g., 224x224)
    3. Convert to tensor
    4. Normalize (mean, std)
    5. Add batch dimension

    Args:
        file: Image file object

    Returns:
        Preprocessed tensor ready for model input
    """
    # TODO: Implement image preprocessing
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img = Image.open(file).convert('RGB')
    tensor = transform(img)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    return tensor
    


# ============================================================================
# FLASK APPLICATION
# ============================================================================

# TODO: Initialize Flask app
app = Flask(__name__)

# TODO: Initialize ModelManager (global singleton)
model_manager = ModelManager(
    mlflow_uri=MLFLOW_TRACKING_URI,
    model_name=MODEL_NAME,
    model_version=MODEL_VERSION
)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint for Kubernetes liveness/readiness probes.

    Returns:
        - 200 if service is healthy
        - 503 if service is not ready (e.g., model not loaded)
    """
    # TODO: Implement health check
    # 
    try:
        model_info = model_manager.get_info()
    
        if model_info.get('status') != 'loaded':
            return jsonify({
                'status': 'unhealthy',
                'reason': 'Model not loaded'
            }), 503
    
        return jsonify({
            'status': 'healthy',
            'model': model_info
        }), 200
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'reason': str(e)
        }), 503
    


@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    """
    Prediction endpoint.

    Expects:
        - Multipart form data with 'file' field (image)
        - X-API-Key header for authentication

    Returns:
        JSON with predictions, confidence scores, and metadata
    """
    # TODO: Implement prediction endpoint
    # Example:
    start_time = time.time()
    #
    try:
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
    
        file = request.files['file']
    
        # Validate file
        validate_image_upload(file)
    
        # Preprocess
        input_tensor = preprocess_image(file)
    
        # Run inference
        result = model_manager.predict(input_tensor)
    
        # Format response
        response = {
            'predictions': result['predictions'],
            'model_version': result['model_version'],
            'inference_time_ms': result['latency_ms']
        }
    
        # Record metrics
        total_latency = time.time() - start_time
        request_latency.labels(
            method='POST',
            endpoint='/predict'
        ).observe(total_latency)
    
        request_count.labels(
            method='POST',
            endpoint='/predict',
            status=200
        ).inc()
    
        logger.info(f"Prediction successful: {total_latency*1000:.2f}ms")
    
        return jsonify(response), 200
    
    except ValueError as e:
        # Validation error
        logger.warning(f"Validation error: {e}")
        request_count.labels(
            method='POST',
            endpoint='/predict',
            status=400
        ).inc()
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        # Server error
        logger.error(f"Prediction error: {e}")
        request_count.labels(
            method='POST',
            endpoint='/predict',
            status=500
        ).inc()
        return jsonify({'error': 'Internal server error'}), 500
    


@app.route('/info', methods=['GET'])
@require_api_key
def info():
    """
    Get model and service information.

    Returns:
        JSON with model version, name, and service metadata
    """
    # TODO: Implement info endpoint
    # Example:
    try:
        model_info = model_manager.get_info()
    
        return jsonify({
            'service': 'ml-api',
            'version': '1.0.0',
            'model': model_info
        }), 200
    
    except Exception as e:
        logger.error(f"Info endpoint error: {e}")
        return jsonify({'error': str(e)}), 500
    


@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus-formatted metrics for scraping
    """
    # TODO: Return Prometheus metrics
    # Example:
    return Response(generate_latest(), mimetype='text/plain')
    


@app.route('/reload', methods=['POST'])
@require_api_key
def reload_model():
    """
    Reload model from MLflow (useful for hot-swapping models).

    Requires admin API key.
    """
    # TODO: Implement model reload endpoint
    # Example:
    try:
        logger.info("Reloading model...")
        model_manager.load_model()
    
        return jsonify({
            'status': 'success',
            'message': 'Model reloaded',
            'model': model_manager.get_info()
        }), 200
    
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    


# ============================================================================
# APPLICATION STARTUP
# ============================================================================

#@app.before_first_request
def startup():
    """
    Run startup tasks before handling first request.
    """
    # TODO: Add startup logic
    # Example:
    logger.info("Application starting up...")
    logger.info(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Model: {MODEL_NAME} v{MODEL_VERSION}")
    


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # TODO: Run the application
    # Example:
    #
    logger.info("="*60)
    logger.info("Starting Production ML API Server")
    logger.info("="*60)
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Model: {MODEL_NAME} v{MODEL_VERSION}")
    logger.info(f"API Keys configured: {len(API_KEYS)}")
    logger.info(f"Log level: {LOG_LEVEL}")
    logger.info("="*60)
    # For development (not production!)
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False  # Never use debug=True in production!
    )
    
    # For production, use Gunicorn:
    # gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 main:app
    


# ============================================================================
# STUDENT IMPLEMENTATION CHECKLIST
# ============================================================================

"""
TODO: Implement the following components:

1. Configuration Management:
   [ ] Load all config from environment variables
   [ ] Set up structured logging (JSON format recommended)
   [ ] Configure log levels appropriately

2. Prometheus Metrics:
   [ ] Define all required metrics (counters, histograms, gauges)
   [ ] Instrument all endpoints
   [ ] Track model-specific metrics

3. ModelManager Class:
   [ ] Implement model loading from MLflow
   [ ] Add model caching
   [ ] Implement predict() method
   [ ] Handle model versioning
   [ ] Add error handling and logging

4. Authentication:
   [ ] Implement API key validation decorator
   [ ] Load API keys from Kubernetes Secret
   [ ] Add different permission levels (optional)

5. Input Validation:
   [ ] Validate file uploads (type, size)
   [ ] Verify image files
   [ ] Add comprehensive error messages

6. Preprocessing:
   [ ] Implement image preprocessing pipeline
   [ ] Match preprocessing to model requirements
   [ ] Handle different image formats

7. API Endpoints:
   [ ] Implement /health endpoint
   [ ] Implement /predict endpoint
   [ ] Implement /info endpoint
   [ ] Implement /metrics endpoint
   [ ] Implement /reload endpoint (optional)

8. Error Handling:
   [ ] Add try-except blocks for all operations
   [ ] Return appropriate HTTP status codes
   [ ] Log all errors with context
   [ ] Don't leak sensitive info in error messages

9. Performance:
   [ ] Optimize model loading (cache in memory)
   [ ] Use batch processing if applicable
   [ ] Minimize preprocessing overhead

10. Testing:
    [ ] Write unit tests for each function
    [ ] Write integration tests for endpoints
    [ ] Test error handling
    [ ] Test with various inputs

11. Documentation:
    [ ] Add docstrings to all functions
    [ ] Document configuration options
    [ ] Create API documentation (OpenAPI/Swagger)
    [ ] Write deployment guide

12. Production Readiness:
    [ ] Remove all debug code
    [ ] Set appropriate timeouts
    [ ] Configure Gunicorn for production
    [ ] Add graceful shutdown handling
    [ ] Test under load

Estimated Time: 40-50 hours for complete implementation
"""
