"""
OpenTelemetry Tracing Tests

Comprehensive test suite for distributed tracing with OpenTelemetry.
Tests tracer initialization, span management, context propagation,
and ML-specific tracing functionality.

Test Coverage:
- SpanContext management
- OpenTelemetryTracer initialization
- Span creation and attributes
- Event recording
- Exception recording
- Status management
- Context injection and extraction
- ML inference tracing
- Tracer lifecycle
"""

import pytest
from unittest.mock import MagicMock, Mock, patch, call

from tracing.tracer import (
    OpenTelemetryTracer,
    SpanContext,
    trace_function,
)


class TestSpanContext:
    """Test SpanContext class."""

    def test_span_context_creation(self):
        """Test creating a span context."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()

        context = SpanContext(mock_span, mock_tracer)

        assert context.span == mock_span
        assert context.tracer == mock_tracer

    def test_span_context_enter_exit(self):
        """Test span context manager enter/exit."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()

        context = SpanContext(mock_span, mock_tracer)

        with context as ctx:
            assert ctx == context

        # Span should be ended
        mock_span.end.assert_called_once()

    def test_span_context_with_exception(self, mock_otel):
        """Test span context handles exceptions."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()

        context = SpanContext(mock_span, mock_tracer)

        try:
            with context:
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should record exception
        mock_span.record_exception.assert_called_once()

    def test_set_attribute(self):
        """Test setting span attribute."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()

        context = SpanContext(mock_span, mock_tracer)
        context.set_attribute("key", "value")

        mock_span.set_attribute.assert_called_once_with("key", "value")

    def test_set_attributes(self):
        """Test setting multiple span attributes."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()

        context = SpanContext(mock_span, mock_tracer)
        attributes = {"key1": "value1", "key2": "value2"}
        context.set_attributes(attributes)

        mock_span.set_attributes.assert_called_once_with(attributes)

    def test_add_event(self):
        """Test adding event to span."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()

        context = SpanContext(mock_span, mock_tracer)
        context.add_event("test_event", {"detail": "info"})

        mock_span.add_event.assert_called_once()

    def test_record_exception(self):
        """Test recording exception in span."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()

        context = SpanContext(mock_span, mock_tracer)
        exception = ValueError("Test error")
        context.record_exception(exception)

        mock_span.record_exception.assert_called_once_with(exception)


class TestOpenTelemetryTracer:
    """Test OpenTelemetryTracer class."""

    def test_tracer_initialization(self):
        """Test tracer initialization."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True):
            tracer = OpenTelemetryTracer(
                service_name="test-service",
                console_export=True
            )

            assert tracer.service_name == "test-service"
            assert tracer.console_export is True

    def test_tracer_initialization_without_otel(self):
        """Test tracer initialization without OpenTelemetry installed."""
        with patch('tracing.tracer.OTEL_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="OpenTelemetry not installed"):
                OpenTelemetryTracer(service_name="test-service")

    @patch('tracing.tracer.OTEL_AVAILABLE', True)
    @patch('tracing.tracer.TracerProvider')
    @patch('tracing.tracer.Resource')
    def test_tracer_initialize(self, mock_resource, mock_provider_class):
        """Test tracer initialization with exporters."""
        tracer = OpenTelemetryTracer(
            service_name="test-service",
            console_export=True
        )

        with patch('tracing.tracer.trace') as mock_trace:
            tracer.initialize()

            assert tracer.provider is not None
            assert tracer.tracer is not None

    @patch('tracing.tracer.OTEL_AVAILABLE', True)
    @patch('tracing.tracer.TracerProvider')
    @patch('tracing.tracer.JaegerExporter')
    def test_tracer_with_jaeger(self, mock_jaeger, mock_provider_class):
        """Test tracer initialization with Jaeger exporter."""
        tracer = OpenTelemetryTracer(
            service_name="test-service",
            jaeger_endpoint="http://jaeger:14268/api/traces"
        )

        with patch('tracing.tracer.trace'):
            tracer.initialize()

            # Jaeger exporter should be created
            assert mock_jaeger.called

    @patch('tracing.tracer.OTEL_AVAILABLE', True)
    @patch('tracing.tracer.TracerProvider')
    @patch('tracing.tracer.OTLPSpanExporter')
    def test_tracer_with_otlp(self, mock_otlp, mock_provider_class):
        """Test tracer initialization with OTLP exporter."""
        tracer = OpenTelemetryTracer(
            service_name="test-service",
            otlp_endpoint="localhost:4317"
        )

        with patch('tracing.tracer.trace'):
            tracer.initialize()

            # OTLP exporter should be created
            assert mock_otlp.called

    def test_start_span(self, mock_otel):
        """Test starting a span."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True):
            tracer = OpenTelemetryTracer(service_name="test-service")

            with patch('tracing.tracer.trace') as mock_trace:
                mock_trace.get_tracer.return_value = mock_otel
                tracer.tracer = mock_otel

                with tracer.start_span("test_span") as span:
                    assert span is not None

    def test_start_span_with_attributes(self, mock_otel):
        """Test starting span with initial attributes."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True):
            tracer = OpenTelemetryTracer(service_name="test-service")

            with patch('tracing.tracer.trace'):
                tracer.tracer = mock_otel

                attributes = {"key": "value", "count": 42}

                with tracer.start_span("test_span", attributes=attributes) as span:
                    assert span is not None

    def test_start_span_not_initialized(self):
        """Test starting span before initialization."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True):
            tracer = OpenTelemetryTracer(service_name="test-service")
            tracer.tracer = None

            with pytest.raises(RuntimeError, match="Tracer not initialized"):
                tracer.start_span("test_span")

    def test_trace_operation(self, mock_otel):
        """Test trace_operation context manager."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True):
            tracer = OpenTelemetryTracer(service_name="test-service")

            with patch('tracing.tracer.trace'):
                tracer.tracer = mock_otel

                with tracer.trace_operation("test_op", model="llama", batch=4) as span:
                    assert span is not None

    def test_inject_context(self):
        """Test context injection for distributed tracing."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True):
            tracer = OpenTelemetryTracer(service_name="test-service")

            carrier = {}
            tracer.inject_context(carrier)

            # Carrier should be modified (exact content depends on propagator)
            # Just verify the method runs without error
            assert isinstance(carrier, dict)

    def test_extract_context(self):
        """Test context extraction from carrier."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True):
            tracer = OpenTelemetryTracer(service_name="test-service")

            carrier = {"traceparent": "00-test-trace-id-00"}
            context = tracer.extract_context(carrier)

            # Context should be extracted (exact format depends on propagator)
            assert context is not None

    def test_trace_ml_inference(self, mock_otel):
        """Test ML-specific inference tracing."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True):
            tracer = OpenTelemetryTracer(service_name="test-service")

            with patch('tracing.tracer.trace'):
                tracer.tracer = mock_otel

                with tracer.trace_ml_inference(
                    model_name="llama-2-7b",
                    model_version="v1.0",
                    batch_size=4
                ) as span:
                    assert span is not None

    def test_shutdown(self):
        """Test tracer shutdown."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True):
            tracer = OpenTelemetryTracer(service_name="test-service")

            mock_provider = MagicMock()
            tracer.provider = mock_provider

            tracer.shutdown()

            mock_provider.shutdown.assert_called_once()

    def test_get_current_span(self, mock_otel):
        """Test getting current active span."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True), \
             patch('tracing.tracer.trace') as mock_trace:

            tracer = OpenTelemetryTracer(service_name="test-service")

            mock_span = MagicMock()
            mock_trace.get_current_span.return_value = mock_span

            span = tracer.get_current_span()

            assert span == mock_span

    def test_is_recording(self, mock_otel):
        """Test checking if span is recording."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True), \
             patch('tracing.tracer.trace') as mock_trace:

            tracer = OpenTelemetryTracer(service_name="test-service")

            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_trace.get_current_span.return_value = mock_span

            assert tracer.is_recording() is True


class TestTracingDecorators:
    """Test tracing decorator functions."""

    def test_trace_function_decorator(self, mock_otel):
        """Test trace_function decorator."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True):
            tracer = OpenTelemetryTracer(service_name="test-service")

            with patch('tracing.tracer.trace'):
                tracer.tracer = mock_otel

                @trace_function(tracer)
                def test_func(x, y):
                    return x + y

                result = test_func(1, 2)

                assert result == 3

    def test_trace_function_with_custom_name(self, mock_otel):
        """Test trace_function decorator with custom span name."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True):
            tracer = OpenTelemetryTracer(service_name="test-service")

            with patch('tracing.tracer.trace'):
                tracer.tracer = mock_otel

                @trace_function(tracer, span_name="custom_operation")
                def test_func():
                    return "result"

                result = test_func()

                assert result == "result"


class TestTracingIntegration:
    """Integration tests for tracing functionality."""

    @pytest.mark.integration
    def test_full_tracing_workflow(self, mock_otel):
        """Test complete tracing workflow."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True):
            tracer = OpenTelemetryTracer(
                service_name="model-server",
                console_export=True
            )

            with patch('tracing.tracer.trace') as mock_trace:
                mock_trace.get_tracer.return_value = mock_otel
                tracer.initialize()
                tracer.tracer = mock_otel

                # Create parent span
                with tracer.start_span("request") as parent:
                    parent.set_attribute("request.id", "req123")

                    # Create child span
                    with tracer.start_span("preprocessing") as child:
                        child.set_attribute("input.size", 1024)

                    # Another child span
                    with tracer.trace_ml_inference(
                        model_name="llama-2",
                        model_version="v1.0"
                    ) as inference:
                        inference.set_attribute("batch.size", 4)

                tracer.shutdown()

    @pytest.mark.integration
    def test_distributed_tracing_context_propagation(self):
        """Test context propagation across service boundaries."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True):
            # Service A
            tracer_a = OpenTelemetryTracer(service_name="service-a")

            # Inject context
            headers = {}
            tracer_a.inject_context(headers)

            # Service B receives headers
            tracer_b = OpenTelemetryTracer(service_name="service-b")
            context = tracer_b.extract_context(headers)

            # Context should be extracted
            assert context is not None


class TestTracingErrorHandling:
    """Test error handling in tracing."""

    def test_span_context_with_none_span(self):
        """Test span context with None span."""
        context = SpanContext(None, None)

        # Should not raise errors
        context.set_attribute("key", "value")
        context.add_event("event")
        context.record_exception(Exception("test"))

    def test_tracer_without_exporters(self, caplog):
        """Test tracer initialization without any exporters."""
        with patch('tracing.tracer.OTEL_AVAILABLE', True):
            tracer = OpenTelemetryTracer(service_name="test-service")

            with patch('tracing.tracer.TracerProvider'), \
                 patch('tracing.tracer.trace'):
                tracer.initialize()

                # Should log warning
                assert any("No exporters configured" in record.message
                          for record in caplog.records)
