"""
Tracing Module - Distributed Tracing for Model Serving

This module provides OpenTelemetry-based distributed tracing for tracking
inference requests across the model serving infrastructure. Includes support
for Jaeger and OTLP exporters, automatic FastAPI middleware, and ML-specific
span attributes.

Components:
    - OpenTelemetryTracer: Core tracing functionality
    - TracingMiddleware: FastAPI middleware for automatic tracing
    - SpanContext: Context manager for creating spans

Example:
    ```python
    from tracing import OpenTelemetryTracer, TracingMiddleware
    from fastapi import FastAPI

    # Initialize tracer
    tracer = OpenTelemetryTracer(
        service_name="model-server",
        jaeger_endpoint="http://jaeger:14268/api/traces"
    )
    tracer.initialize()

    # Add to FastAPI app
    app = FastAPI()
    app.add_middleware(TracingMiddleware, tracer=tracer)

    # Manual tracing
    with tracer.start_span("inference") as span:
        span.set_attribute("model", "llama-2-7b")
        result = model.generate(prompt)
        span.set_attribute("tokens", len(result))
    ```
"""

from .tracer import OpenTelemetryTracer, SpanContext
from .middleware import TracingMiddleware

__all__ = [
    "OpenTelemetryTracer",
    "SpanContext",
    "TracingMiddleware",
]

__version__ = "1.0.0"
