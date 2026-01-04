"""
ML API with Advanced Custom Metrics

This version integrates drift detection, performance monitoring, and quality checks.
"""

from flask import Flask, jsonify, request
import logging
import json
import time
import random
from datetime import datetime
import os
import numpy as np

# Import instrumentation
from instrumentation import (
    MetricsMiddleware,
    SystemMetricsCollector,
    metrics_endpoint
)

# Import custom metrics
from custom_metrics import (
    DataDriftDetector,
    ModelPerformanceMonitor,
    ConfidenceAnalyzer,
    DataQualityMonitor
)

# Configure structured JSON logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
        if hasattr(record, 'status_code'):
            log_data['status_code'] = record.status_code
        if hasattr(record, 'endpoint'):
            log_data['endpoint'] = record.endpoint
            
        return json.dumps(log_data)

# Setup logging
log_dir = '/var/log/app'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'ml-api.log')),
        logging.StreamHandler()
    ]
)

for handler in logging.root.handlers:
    handler.setFormatter(JSONFormatter())

logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Initialize metrics middleware
metrics_middleware = MetricsMiddleware(app)

# Start system metrics collection
collector = SystemMetricsCollector(interval=15)
collector.start_background_collection()

# Initialize custom metrics components
# Create reference data for drift detection (simulated training data)
np.random.seed(42)
reference_data = np.random.normal(loc=2.0, scale=1.0, size=(1000, 3))
feature_names = ['feature1', 'feature2', 'feature3']

drift_detector = DataDriftDetector(
    reference_data=reference_data,
    feature_names=feature_names,
    threshold=0.05,
    method='ks'
)

# Initialize performance monitor
performance_monitor = ModelPerformanceMonitor(
    model_name='demo-model',
    min_samples=10
)

# Initialize confidence analyzer
confidence_analyzer = ConfidenceAnalyzer(window_size=1000)

# Initialize data quality monitor
data_quality_monitor = DataQualityMonitor(
    expected_schema={
        'feature1': 'float',
        'feature2': 'float',
        'feature3': 'float'
    }
)

# Set initial model accuracy
metrics_middleware.update_model_accuracy('demo-model', 0.92)

# =============================================================================
# API Endpoints
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Prometheus metrics endpoint."""
    return metrics_endpoint()

@app.route('/predict', methods=['POST'])
def predict():
    """
    ML prediction endpoint with advanced monitoring.
    
    Expected JSON body:
    {
        "features": {
            "feature1": 1.0,
            "feature2": 2.0,
            "feature3": 3.0
        }
    }
    """
    start_time = time.time()
    
    try:
        # Validate request
        if not request.is_json:
            logger.warning("Invalid request: not JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        if 'features' not in data:
            logger.warning("Invalid request: missing features")
            return jsonify({'error': 'Missing features'}), 400
        
        features = data['features']
        
        # Data quality validation
        quality_issues = data_quality_monitor.validate_request(features)
        if quality_issues['missing']:
            metrics_middleware.track_data_quality(
                {feat: 1 for feat in quality_issues['missing']}
            )
            logger.warning(f"Missing features: {quality_issues['missing']}")
        
        # Extract feature values
        feature_values = np.array([
            features.get('feature1', 0.0),
            features.get('feature2', 0.0),
            features.get('feature3', 0.0)
        ]).reshape(1, 3)
        
        # Check for data drift periodically
        if random.random() < 0.1:  # Check 10% of requests to reduce overhead
            try:
                drift_results = drift_detector.detect_drift(feature_values)
                drift_detector.export_drift_metrics(drift_results)
                
                # Alert if drift detected
                drifted_features = [r.feature_name for r in drift_results if r.is_drift]
                if drifted_features:
                    logger.warning(f"Data drift detected in features: {drifted_features}")
            except Exception as e:
                logger.error(f"Drift detection failed: {e}")
        
        # Simulate model inference
        inference_start = time.time()
        time.sleep(random.uniform(0.01, 0.1))
        
        # Generate prediction
        classes = ['cat', 'dog', 'bird']
        prediction_class = random.choice(classes)
        confidence = random.uniform(0.7, 0.99)
        
        inference_time = time.time() - inference_start
        
        # Track prediction metrics
        metrics_middleware.track_prediction(
            model_name='demo-model',
            prediction_class=prediction_class,
            confidence=confidence,
            inference_time=inference_time
        )
        
        # Log confidence
        confidence_analyzer.log_confidence(confidence)
        
        # Simulate ground truth feedback (in production, this comes later)
        # For demo: 90% accuracy
        if random.random() < 0.9:
            ground_truth = prediction_class
        else:
            ground_truth = random.choice([c for c in classes if c != prediction_class])
        
        # Log prediction for performance monitoring
        performance_monitor.log_prediction(
            prediction=prediction_class,
            ground_truth=ground_truth
        )
        
        # Periodically calculate performance metrics
        if len(performance_monitor.ground_truth) >= performance_monitor.min_samples:
            if random.random() < 0.05:  # Calculate 5% of the time
                perf_metrics = performance_monitor.calculate_metrics()
                if perf_metrics:
                    # Check for degradation
                    is_degraded = performance_monitor.check_degradation(
                        baseline_accuracy=0.92,
                        threshold=0.1
                    )
                    if is_degraded:
                        logger.error("Model performance has degraded!")
        
        total_time = time.time() - start_time
        
        # Log prediction
        logger.info(
            "Prediction completed",
            extra={
                'endpoint': '/predict',
                'status_code': 200,
                'duration_ms': total_time * 1000,
                'prediction_class': prediction_class,
                'confidence': confidence,
                'inference_time_ms': inference_time * 1000
            }
        )
        
        return jsonify({
            'prediction': prediction_class,
            'confidence': confidence,
            'inference_time_ms': inference_time * 1000,
            'ground_truth': ground_truth  # Remove in production!
        })
    
    except Exception as e:
        logger.error(
            f"Prediction failed: {str(e)}",
            extra={
                'endpoint': '/predict',
                'status_code': 500,
                'error': str(e)
            },
            exc_info=True
        )
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Get monitoring statistics."""
    try:
        confidence_stats = confidence_analyzer.get_statistics()
        
        return jsonify({
            'confidence_analysis': confidence_stats,
            'total_predictions': len(performance_monitor.predictions),
            'predictions_with_ground_truth': len(performance_monitor.ground_truth)
        })
    except Exception as e:
        logger.error(f"Stats endpoint failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/drift', methods=['POST'])
def check_drift():
    """Manually trigger drift check on batch of data."""
    try:
        data = request.get_json()
        
        if 'samples' not in data:
            return jsonify({'error': 'Missing samples array'}), 400
        
        # Convert to numpy array
        samples = np.array(data['samples'])
        
        # Detect drift
        drift_results = drift_detector.detect_drift(samples)
        drift_detector.export_drift_metrics(drift_results)
        
        # Format results
        results = []
        for r in drift_results:
            results.append({
                'feature': r.feature_name,
                'is_drift': r.is_drift,
                'statistic': r.statistic,
                'p_value': r.p_value,
                'method': r.test_method
            })
        
        return jsonify({'drift_results': results})
        
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/error', methods=['GET'])
def trigger_error():
    """Endpoint to trigger an error for testing alerts."""
    logger.error("Test error triggered")
    return jsonify({'error': 'Test error'}), 500

@app.route('/slow', methods=['GET'])
def slow_endpoint():
    """Endpoint that's intentionally slow for testing latency alerts."""
    time.sleep(2)
    return jsonify({'message': 'This was slow'})

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    logger.info("Starting ML API server with custom metrics")
    app.run(host='0.0.0.0', port=5000, debug=False)