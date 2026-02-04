"""
Tracing Middleware for FastAPI

Automatic request tracing middleware that instruments FastAPI applications
with distributed tracing support.
"""

import logging
import time
import uuid
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

try:
    from opentelemetry.trace import StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

logger = logging.getLogger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic request tracing.

    This middleware automatically creates trace spans for all incoming HTTP
    requests, capturing request/response metadata, latency, and errors.
    It also extracts and propagates trace context across service boundaries.

    Attributes:
        tracer: OpenTelemetryTracer instance
        exclude_paths: List of paths to exclude from tracing
        capture_headers: Whether to capture request/response headers
        capture_body: Whether to capture request/response bodies

    Example:
        ```python
        from fastapi import FastAPI
        from tracing import OpenTelemetryTracer, TracingMiddleware

        tracer = OpenTelemetryTracer(
            service_name="model-api",
            jaeger_endpoint="http://jaeger:14268/api/traces"
        )
        tracer.initialize()

        app = FastAPI()
        app.add_middleware(
            TracingMiddleware,
            tracer=tracer,
            exclude_paths=["/health", "/metrics"]
        )

        @app.get("/generate")
        async def generate(prompt: str):
            # Automatically traced
            return {"response": model.generate(prompt)}
        ```
    """

    def __init__(
        self,
        app,
        tracer,
        exclude_paths: Optional[list] = None,
        capture_headers: bool = True,
        capture_body: bool = False,
    ):
        """
        Initialize tracing middleware.

        Args:
            app: FastAPI application
            tracer: OpenTelemetryTracer instance
            exclude_paths: Paths to exclude from tracing (e.g., /health)
            capture_headers: Capture request/response headers
            capture_body: Capture request/response bodies (may be large)
        """
        super().__init__(app)
        self.tracer = tracer
        self.exclude_paths = exclude_paths or []
        self.capture_headers = capture_headers
        self.capture_body = capture_body

        logger.info(
            f"TracingMiddleware initialized with {len(self.exclude_paths)} excluded paths"
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process incoming request and create trace span.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response
        """
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Extract trace context from headers
        context = self.tracer.extract_context(dict(request.headers))

        # Create span for request
        span_name = f"{request.method} {request.url.path}"

        with self.tracer.start_span(span_name, parent=context) as span:
            # Set request attributes
            self._set_request_attributes(span, request, request_id)

            # Record start time
            start_time = time.time()

            try:
                # Process request
                response = await call_next(request)

                # Calculate latency
                latency = time.time() - start_time

                # Set response attributes
                self._set_response_attributes(span, response, latency)

                # Set status based on HTTP status code
                if 200 <= response.status_code < 400:
                    if OTEL_AVAILABLE:
                        span.set_status(StatusCode.OK)
                else:
                    if OTEL_AVAILABLE:
                        span.set_status(
                            StatusCode.ERROR,
                            f"HTTP {response.status_code}"
                        )

                # Add request ID to response headers
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Trace-ID"] = self._get_trace_id(span)

                return response

            except Exception as e:
                # Record exception
                latency = time.time() - start_time
                span.record_exception(e)

                if OTEL_AVAILABLE:
                    span.set_status(StatusCode.ERROR, str(e))

                span.set_attribute("http.error", str(e))
                span.set_attribute("http.latency_ms", latency * 1000)

                logger.error(
                    f"Request {request_id} failed: {e}",
                    exc_info=True,
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                    }
                )

                # Re-raise exception
                raise

    def _set_request_attributes(
        self,
        span,
        request: Request,
        request_id: str
    ) -> None:
        """
        Set request-related span attributes.

        Args:
            span: SpanContext instance
            request: HTTP request
            request_id: Generated request ID
        """
        # Standard HTTP attributes
        span.set_attributes({
            "http.method": request.method,
            "http.url": str(request.url),
            "http.path": request.url.path,
            "http.scheme": request.url.scheme,
            "http.host": request.url.hostname or "",
            "http.request_id": request_id,
        })

        # Query parameters
        if request.query_params:
            for key, value in request.query_params.items():
                span.set_attribute(f"http.query.{key}", value)

        # Client information
        if request.client:
            span.set_attribute("http.client.ip", request.client.host)
            span.set_attribute("http.client.port", request.client.port)

        # User agent
        user_agent = request.headers.get("user-agent", "")
        if user_agent:
            span.set_attribute("http.user_agent", user_agent)

        # Request headers (if enabled)
        if self.capture_headers:
            for key, value in request.headers.items():
                # Skip sensitive headers
                if key.lower() not in ["authorization", "cookie", "x-api-key"]:
                    span.set_attribute(f"http.request.header.{key}", value)

        # Add event for request received
        span.add_event("request.received", {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        })

    def _set_response_attributes(
        self,
        span,
        response: Response,
        latency: float
    ) -> None:
        """
        Set response-related span attributes.

        Args:
            span: SpanContext instance
            response: HTTP response
            latency: Request latency in seconds
        """
        # Standard response attributes
        span.set_attributes({
            "http.status_code": response.status_code,
            "http.latency_ms": latency * 1000,
            "http.latency_seconds": latency,
        })

        # Response headers (if enabled)
        if self.capture_headers:
            for key, value in response.headers.items():
                # Skip sensitive headers
                if key.lower() not in ["set-cookie"]:
                    span.set_attribute(f"http.response.header.{key}", value)

        # Content type and length
        content_type = response.headers.get("content-type", "")
        if content_type:
            span.set_attribute("http.response.content_type", content_type)

        content_length = response.headers.get("content-length", "")
        if content_length:
            span.set_attribute("http.response.content_length", int(content_length))

        # Classify latency
        latency_class = self._classify_latency(latency)
        span.set_attribute("http.latency_class", latency_class)

        # Add event for response sent
        span.add_event("response.sent", {
            "status_code": response.status_code,
            "latency_ms": latency * 1000,
        })

    def _classify_latency(self, latency: float) -> str:
        """
        Classify latency into categories.

        Args:
            latency: Latency in seconds

        Returns:
            Latency classification
        """
        if latency < 0.1:
            return "fast"
        elif latency < 1.0:
            return "normal"
        elif latency < 5.0:
            return "slow"
        else:
            return "very_slow"

    def _get_trace_id(self, span) -> str:
        """
        Extract trace ID from span.

        Args:
            span: SpanContext instance

        Returns:
            Trace ID as hex string
        """
        try:
            if hasattr(span, 'span') and hasattr(span.span, 'get_span_context'):
                context = span.span.get_span_context()
                if hasattr(context, 'trace_id'):
                    return format(context.trace_id, '032x')
        except Exception:
            pass

        return "unknown"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding request IDs to all requests.

    This is a lightweight alternative to full tracing when you only
    need request identification.

    Example:
        ```python
        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)

        @app.get("/test")
        async def test(request: Request):
            request_id = request.state.request_id
            logger.info(f"Processing request {request_id}")
            return {"request_id": request_id}
        ```
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Add request ID to request and response.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response with X-Request-ID header
        """
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Store in request state
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add to response headers
        response.headers["X-Request-ID"] = request_id

        return response
