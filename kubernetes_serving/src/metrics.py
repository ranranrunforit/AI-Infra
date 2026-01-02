
"""
Prometheus metrics definitions.

This module contains all the metric objects used to track application performance.
Separating them here allows them to be imported by multiple modules if needed.
"""

from prometheus_client import Counter, Histogram, Gauge
# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# TODO: Define Prometheus metrics using prometheus_client
# These metrics will be scraped by Prometheus every 30 seconds

# Counter: Monotonically increasing value (requests, errors, predictions)
# Histogram: Distribution of values (latency, inference time)
# Gauge: Value that can go up or down (model loaded status, active connections)

# TODO: Create request counter
# Track total requests with labels: method, endpoint, status_code
request_count = Counter(
    'model_api_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status_code']
)

# TODO: Create request duration histogram
# Track request latency with labels: method, endpoint
# Buckets: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0] (in seconds)
request_duration = Histogram(
    'model_api_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# TODO: Create prediction counter
# Track total predictions with labels: model_name, status (success/error)
prediction_count = Counter(
    'model_api_predictions_total',
    'Total number of predictions',
    ['model_name', 'status']
)

# TODO: Create inference duration histogram
# Track model inference time with labels: model_name
# Buckets: [0.01, 0.05, 0.1, 0.2, 0.5, 1.0] (in seconds)
inference_duration = Histogram(
    'model_api_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model_name'],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
)

# TODO: Create model loaded gauge
# Value: 1 if model loaded, 0 if not
# Labels: model_name, version
model_loaded_gauge = Gauge(
    'model_api_model_loaded',
    'Whether model is loaded (1=loaded, 0=not loaded)',
    ['model_name', 'version']
)

# TODO: Create active connections gauge
# Track current number of active requests
active_connections = Gauge(
    'model_api_active_connections',
    'Number of active connections'
)
