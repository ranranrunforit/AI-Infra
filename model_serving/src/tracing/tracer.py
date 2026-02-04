"""
OpenTelemetry Tracer Module

Production-ready distributed tracing with Jaeger and OTLP support for
ML inference workloads.
"""

import logging
from typing import Optional, Dict, Any, ContextManager
from contextlib import contextmanager

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning(
        "OpenTelemetry not installed. Install with: "
        "pip install opentelemetry-api opentelemetry-sdk "
        "opentelemetry-exporter-jaeger opentelemetry-exporter-otlp"
    )

logger = logging.getLogger(__name__)


class SpanContext:
    """
    Context manager for creating and managing trace spans.

    This class provides a convenient way to create spans with automatic
    error handling and span finalization.
    """

    def __init__(self, span, tracer_instance):
        """
        Initialize span context.

        Args:
            span: OpenTelemetry span object
            tracer_instance: Parent tracer instance
        """
        self.span = span
        self.tracer = tracer_instance

    def __enter__(self):
        """Enter span context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit span context and record any exceptions."""
        if exc_type is not None:
            self.record_exception(exc_val)
            self.set_status(StatusCode.ERROR, str(exc_val))
        else:
            self.set_status(StatusCode.OK)

        self.span.end()
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        """
        Set a span attribute.

        Args:
            key: Attribute key
            value: Attribute value
        """
        if self.span:
            self.span.set_attribute(key, value)

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """
        Set multiple span attributes.

        Args:
            attributes: Dictionary of attributes
        """
        if self.span:
            self.span.set_attributes(attributes)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an event to the span.

        Args:
            name: Event name
            attributes: Event attributes
        """
        if self.span:
            self.span.add_event(name, attributes=attributes or {})

    def record_exception(self, exception: Exception) -> None:
        """
        Record an exception in the span.

        Args:
            exception: Exception to record
        """
        if self.span:
            self.span.record_exception(exception)

    def set_status(self, status_code: StatusCode, description: str = "") -> None:
        """
        Set span status.

        Args:
            status_code: Status code (OK, ERROR, UNSET)
            description: Optional status description
        """
        if self.span:
            if OTEL_AVAILABLE:
                self.span.set_status(Status(status_code, description))


class OpenTelemetryTracer:
    """
    OpenTelemetry tracer for distributed tracing.

    This class provides a high-level interface to OpenTelemetry tracing with
    support for Jaeger and OTLP exporters. It includes ML-specific attributes
    and automatic context propagation.

    Attributes:
        service_name: Name of the service
        tracer: OpenTelemetry tracer instance
        provider: Tracer provider

    Example:
        ```python
        # Initialize tracer
        tracer = OpenTelemetryTracer(
            service_name="model-server",
            jaeger_endpoint="http://jaeger:14268/api/traces"
        )
        tracer.initialize()

        # Create spans
        with tracer.start_span("inference") as span:
            span.set_attribute("model.name", "llama-2-7b")
            span.set_attribute("model.batch_size", 32)
            result = run_inference()
            span.set_attribute("output.tokens", len(result))

        # Inject context for distributed tracing
        headers = {}
        tracer.inject_context(headers)
        # Send headers with HTTP request

        # Extract context from incoming request
        tracer.extract_context(request_headers)
        ```
    """

    def __init__(
        self,
        service_name: str,
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        console_export: bool = False,
        sample_rate: float = 1.0,
    ):
        """
        Initialize OpenTelemetry tracer.

        Args:
            service_name: Name of the service for trace identification
            jaeger_endpoint: Jaeger collector endpoint (e.g., http://localhost:14268/api/traces)
            otlp_endpoint: OTLP gRPC endpoint (e.g., localhost:4317)
            console_export: Export traces to console for debugging
            sample_rate: Sampling rate (0.0-1.0)
        """
        if not OTEL_AVAILABLE:
            raise RuntimeError(
                "OpenTelemetry not installed. Please install with: "
                "pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-jaeger opentelemetry-exporter-otlp"
            )

        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.otlp_endpoint = otlp_endpoint
        self.console_export = console_export
        self.sample_rate = sample_rate

        self.provider: Optional[TracerProvider] = None
        self.tracer = None
        self.propagator = TraceContextTextMapPropagator()

        logger.info(f"OpenTelemetryTracer created for service: {service_name}")

    def initialize(self) -> None:
        """
        Initialize the tracer with configured exporters.

        This method sets up the TracerProvider and configures exporters
        for Jaeger, OTLP, or console output.
        """
        # Create resource with service information
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "1.0.0",
        })

        # Create tracer provider
        self.provider = TracerProvider(resource=resource)

        # Add exporters
        if self.jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                collector_endpoint=self.jaeger_endpoint,
            )
            self.provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
            logger.info(f"Jaeger exporter configured: {self.jaeger_endpoint}")

        if self.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.otlp_endpoint,
                insecure=True,
            )
            self.provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
            logger.info(f"OTLP exporter configured: {self.otlp_endpoint}")

        if self.console_export:
            console_exporter = ConsoleSpanExporter()
            self.provider.add_span_processor(
                BatchSpanProcessor(console_exporter)
            )
            logger.info("Console exporter configured")

        if not (self.jaeger_endpoint or self.otlp_endpoint or self.console_export):
            logger.warning("No exporters configured, traces will not be exported")

        # Set as global tracer provider
        trace.set_tracer_provider(self.provider)

        # Get tracer
        self.tracer = trace.get_tracer(__name__)

        logger.info("OpenTelemetry tracer initialized")

    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[Any] = None,
    ) -> SpanContext:
        """
        Start a new trace span.

        Args:
            name: Span name
            attributes: Initial span attributes
            parent: Parent span context (optional)

        Returns:
            SpanContext that can be used as a context manager

        Example:
            ```python
            with tracer.start_span("model_inference") as span:
                span.set_attribute("model", "llama-2-7b")
                result = model.generate(prompt)
            ```
        """
        if not self.tracer:
            raise RuntimeError("Tracer not initialized. Call initialize() first.")

        # Create span
        if parent:
            context = trace.set_span_in_context(parent)
            span = self.tracer.start_span(name, context=context)
        else:
            span = self.tracer.start_span(name)

        # Set initial attributes
        if attributes:
            span.set_attributes(attributes)

        return SpanContext(span, self)

    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        **attributes
    ) -> ContextManager[SpanContext]:
        """
        Context manager for tracing an operation.

        This is a convenience method that combines span creation and
        attribute setting.

        Args:
            operation_name: Name of the operation
            **attributes: Keyword arguments become span attributes

        Yields:
            SpanContext for the operation

        Example:
            ```python
            with tracer.trace_operation("inference", model="llama", batch_size=4) as span:
                result = run_inference()
                span.set_attribute("tokens", len(result))
            ```
        """
        with self.start_span(operation_name, attributes=attributes) as span:
            yield span

    def inject_context(self, carrier: Dict[str, str]) -> None:
        """
        Inject trace context into a carrier (e.g., HTTP headers).

        This is used for propagating trace context across service boundaries.

        Args:
            carrier: Dictionary to inject context into (modified in-place)

        Example:
            ```python
            headers = {}
            tracer.inject_context(headers)
            # headers now contains traceparent and tracestate
            response = requests.post(url, headers=headers, ...)
            ```
        """
        self.propagator.inject(carrier)

    def extract_context(self, carrier: Dict[str, str]) -> Any:
        """
        Extract trace context from a carrier (e.g., HTTP headers).

        This is used for continuing a trace from an incoming request.

        Args:
            carrier: Dictionary containing trace context

        Returns:
            Context object that can be used as parent for new spans

        Example:
            ```python
            # In request handler
            context = tracer.extract_context(request.headers)
            with tracer.start_span("handle_request", parent=context):
                process_request()
            ```
        """
        return self.propagator.extract(carrier)

    def trace_ml_inference(
        self,
        model_name: str,
        model_version: str = "unknown",
        **kwargs
    ) -> SpanContext:
        """
        Start a span specifically for ML inference operations.

        This is a convenience method that adds ML-specific attributes.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            **kwargs: Additional attributes

        Returns:
            SpanContext for the inference operation

        Example:
            ```python
            with tracer.trace_ml_inference(
                model_name="llama-2-7b",
                model_version="v1.0",
                batch_size=4
            ) as span:
                result = model.generate(prompts)
                span.set_attribute("output.tokens", sum(len(r) for r in result))
            ```
        """
        attributes = {
            "ml.model.name": model_name,
            "ml.model.version": model_version,
            **kwargs
        }

        return self.start_span("ml.inference", attributes=attributes)

    def shutdown(self) -> None:
        """
        Shutdown the tracer and flush any pending spans.

        This should be called before application shutdown to ensure
        all traces are exported.
        """
        if self.provider:
            self.provider.shutdown()
            logger.info("Tracer shutdown complete")

    def get_current_span(self):
        """
        Get the current active span.

        Returns:
            Current span or None if no active span
        """
        return trace.get_current_span()

    def is_recording(self) -> bool:
        """
        Check if there is an active recording span.

        Returns:
            True if recording, False otherwise
        """
        span = self.get_current_span()
        return span is not None and span.is_recording()


# Utility functions for common tracing patterns

def trace_function(tracer: OpenTelemetryTracer, span_name: Optional[str] = None):
    """
    Decorator for automatically tracing function calls.

    Args:
        tracer: OpenTelemetryTracer instance
        span_name: Optional span name (defaults to function name)

    Example:
        ```python
        @trace_function(tracer)
        def process_data(data):
            return transform(data)
        ```
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = span_name or func.__name__
            with tracer.start_span(name) as span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.args_count", len(args))
                return func(*args, **kwargs)
        return wrapper
    return decorator


async def trace_async_function(tracer: OpenTelemetryTracer, span_name: Optional[str] = None):
    """
    Decorator for automatically tracing async function calls.

    Args:
        tracer: OpenTelemetryTracer instance
        span_name: Optional span name (defaults to function name)

    Example:
        ```python
        @trace_async_function(tracer)
        async def process_data(data):
            return await transform_async(data)
        ```
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            name = span_name or func.__name__
            with tracer.start_span(name) as span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.args_count", len(args))
                return await func(*args, **kwargs)
        return wrapper
    return decorator
