"""
Model API Application

This module implements the REST API for model inference using Flask.

Author: AI Infrastructure Curriculum
License: MIT
"""

import io
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional
from PIL import Image

from flask import Flask, request, jsonify, Response

# Import the config instance (not the class)
from config import config
from model_loader import ModelLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================================================================
# FLASK IMPLEMENTATION
# =========================================================================

# Initialize Flask app
app = Flask(__name__)

# Set Flask's built-in max content length
app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE

# Initialize model loader (will be loaded in init_model)
model_loader: Optional[ModelLoader] = None


def init_model():
    """Initialize and load ML model."""
    global model_loader
    try:
        logger.info("Initializing model...")
        model_loader = ModelLoader(
            model_name=config.MODEL_NAME,
            device=config.DEVICE
        )
        model_loader.load()
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    
    Returns:
        JSON response with health status
    """
    is_healthy = model_loader is not None and model_loader.model is not None
    
    if is_healthy:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'model_name': model_loader.model_name,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 200
    else:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'reason': 'Model not loaded',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 503


@app.route('/info', methods=['GET'])
def info():
    """
    Model information endpoint.
    
    Returns:
        JSON response with model and API info
    """
    if model_loader is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    model_info = model_loader.get_model_info()
    
    return jsonify({
        'model': model_info,
        'api': {
            'version': config.API_VERSION,
            'endpoints': ['/predict', '/health', '/info']
        },
        'limits': {
            'max_file_size_mb': config.MAX_FILE_SIZE / (1024 * 1024),
            'max_image_dimension': config.MAX_IMAGE_DIMENSION,
            'timeout_seconds': config.REQUEST_TIMEOUT,
            'default_top_k': config.DEFAULT_TOP_K,
            'max_top_k': config.MAX_TOP_K
        },
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    
    Returns:
        JSON response with predictions or error
    """
    correlation_id = generate_correlation_id()
    start_time = time.time()
    
    try:
        # 1. Validate request has file
        if 'file' not in request.files:
            return format_error_response(
                'MISSING_FILE',
                'No file provided in request',
                correlation_id
            ), 400
        
        file = request.files['file']
        
        # 2. Check file has name
        if file.filename == '':
            return format_error_response(
                'EMPTY_FILENAME',
                'Empty filename',
                correlation_id
            ), 400
        
        # 3. Check file size (read and check length)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
    
        if file_size > config.MAX_FILE_SIZE:
            return format_error_response(
                'FILE_TOO_LARGE',
                f'File size {file_size} exceeds limit {config.MAX_FILE_SIZE}',
                correlation_id
            ), 413
    
        # 4. Load image
        try:
            file_bytes = file.read()
            image = Image.open(io.BytesIO(file_bytes))
        except Exception as e:
            return format_error_response(
                'INVALID_IMAGE_FORMAT',
                f'Could not load image: {str(e)}',
                correlation_id
            ), 400
            
        # try:
        #     file_bytes = file.read()
        #     file_size = len(file_bytes)
            
        #     if file_size > config.MAX_FILE_SIZE:
        #         return format_error_response(
        #             'FILE_TOO_LARGE',
        #             f'File size {file_size} bytes exceeds limit {config.MAX_FILE_SIZE} bytes',
        #             correlation_id
        #         ), 413
            
        #     # Load image from bytes
        #     image = Image.open(io.BytesIO(file_bytes))
        # except Exception as e:
        #     return format_error_response(
        #         'INVALID_IMAGE_FORMAT',
        #         f'Could not load image: {str(e)}',
        #         correlation_id
        #     ), 400
        
        # 5. Validate image
        is_valid, error_msg = model_loader.validate_image(image)
        if not is_valid:
            return format_error_response(
                'INVALID_IMAGE',
                error_msg,
                correlation_id
            ), 400
        
        # 6. Get top_k parameter
        top_k = request.form.get('top_k', config.DEFAULT_TOP_K)
        try:
            top_k = int(top_k)
            if top_k < 1 or top_k > config.MAX_TOP_K:
                return format_error_response(
                    'INVALID_PARAMETER',
                    f'top_k must be between 1 and {config.MAX_TOP_K}',
                    correlation_id
                ), 400
        except ValueError:
            return format_error_response(
                'INVALID_PARAMETER',
                'top_k must be an integer',
                correlation_id
            ), 400
        
        # 7. Generate predictions
        predictions = model_loader.predict(image, top_k=top_k)
        
        # 8. Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # 9. Log request
        logger.info(
            f"Prediction successful: correlation_id={correlation_id}, "
            f"latency={latency_ms:.2f}ms, top_class={predictions[0]['class']}"
        )
        
        # 10. Return response
        return format_success_response(predictions, latency_ms, correlation_id), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return format_error_response(
            'INTERNAL_ERROR',
            'Internal server error',
            correlation_id
        ), 500


# =========================================================================
# Error Handlers
# =========================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle 413 Payload Too Large errors."""
    return jsonify({
        'success': False,
        'error': {
            'code': 'FILE_TOO_LARGE',
            'message': f'File size exceeds maximum allowed size of {Config.MAX_FILE_SIZE / (1024 * 1024):.1f} MB',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': {
            'code': 'NOT_FOUND',
            'message': 'Endpoint not found',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'success': False,
        'error': {
            'code': 'METHOD_NOT_ALLOWED',
            'message': 'HTTP method not allowed for this endpoint',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': {
            'code': 'INTERNAL_ERROR',
            'message': 'Internal server error',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    }), 500


# =========================================================================
# Helper Functions
# =========================================================================

def generate_correlation_id() -> str:
    """
    Generate unique correlation ID for request tracking.
    
    Returns:
        Correlation ID string
    """
    return f"req-{uuid.uuid4().hex[:8]}"


def format_success_response(predictions: list,
                           latency_ms: float,
                           correlation_id: str) -> dict:
    """
    Format successful prediction response.
    
    Args:
        predictions: List of prediction dictionaries
        latency_ms: Request latency in milliseconds
        correlation_id: Request correlation ID
        
    Returns:
        Formatted response dictionary
    """
    return {
        'success': True,
        'predictions': predictions,
        'latency_ms': round(latency_ms, 2),
        'correlation_id': correlation_id,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }


def format_error_response(error_code: str,
                         message: str,
                         correlation_id: str,
                         details: Optional[dict] = None) -> dict:
    """
    Format error response.
    
    Args:
        error_code: Error code (e.g., 'INVALID_IMAGE')
        message: Human-readable error message
        correlation_id: Request correlation ID
        details: Optional additional details
        
    Returns:
        Formatted error response dictionary
    """
    error_response = {
        'success': False,
        'error': {
            'code': error_code,
            'message': message,
            'correlation_id': correlation_id,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    }
    if details:
        error_response['error']['details'] = details
    return error_response


# =========================================================================
# Application Startup
# =========================================================================

if __name__ == '__main__':
    """Run Flask application."""
    try:
        # Initialize model
        init_model()
        
        # Start server
        logger.info(f"Starting server on {config.HOST}:{config.PORT}")
        app.run(
            host=config.HOST,
            port=config.PORT,
            debug=config.DEBUG
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        exit(1)