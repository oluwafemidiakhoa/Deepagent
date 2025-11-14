"""
Observability Module

Provides comprehensive logging, metrics, and tracing for DeepAgent:
- Structured logging with structlog
- Metrics collection and export
- OpenTelemetry tracing for distributed systems
- Performance monitoring

Author: Oluwafemi Idiakhoa
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import time
import json
import os


@dataclass
class MetricEvent:
    """Metric event for tracking agent performance"""
    timestamp: datetime = field(default_factory=datetime.now)
    metric_name: str = ""
    metric_value: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "tags": self.tags,
            "metadata": self.metadata
        }


class StructuredLogger:
    """
    Structured logging using structlog

    Provides JSON-formatted logs with context for production debugging
    """

    def __init__(self, name: str = "deepagent", log_level: str = "INFO"):
        self.name = name
        self.log_level = log_level.upper()
        self.structlog_available = False

        # Try to import structlog
        try:
            import structlog

            structlog.configure(
                processors=[
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.add_logger_name,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer()
                ],
                wrapper_class=structlog.make_filtering_bound_logger(
                    getattr(structlog.stdlib, log_level, structlog.stdlib.INFO)
                ),
                context_class=dict,
                logger_factory=structlog.PrintLoggerFactory(),
                cache_logger_on_first_use=True,
            )

            self.logger = structlog.get_logger(name)
            self.structlog_available = True
            print(f"Structured logging enabled with structlog for {name}")

        except ImportError:
            print("Warning: structlog not installed. Using basic Python logging.")
            print("Install with: pip install structlog")
            import logging
            logging.basicConfig(
                level=getattr(logging, log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(name)

    def debug(self, event: str, **kwargs):
        """Log debug message"""
        if self.structlog_available:
            self.logger.debug(event, **kwargs)
        else:
            self.logger.debug(f"{event} {kwargs}")

    def info(self, event: str, **kwargs):
        """Log info message"""
        if self.structlog_available:
            self.logger.info(event, **kwargs)
        else:
            self.logger.info(f"{event} {kwargs}")

    def warning(self, event: str, **kwargs):
        """Log warning message"""
        if self.structlog_available:
            self.logger.warning(event, **kwargs)
        else:
            self.logger.warning(f"{event} {kwargs}")

    def error(self, event: str, **kwargs):
        """Log error message"""
        if self.structlog_available:
            self.logger.error(event, **kwargs)
        else:
            self.logger.error(f"{event} {kwargs}")

    def bind(self, **kwargs):
        """Bind context to logger"""
        if self.structlog_available:
            return self.logger.bind(**kwargs)
        return self


class MetricsCollector:
    """
    Collects and tracks agent performance metrics

    Tracks:
    - Tool execution times
    - LLM inference latency
    - Memory usage
    - Success/failure rates
    - Custom metrics
    """

    def __init__(self, export_interval: int = 60):
        self.metrics: List[MetricEvent] = []
        self.export_interval = export_interval
        self.last_export_time = time.time()

        # Aggregated stats
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}

    def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric (incremental)"""
        key = self._make_key(name, tags)
        self.counters[key] = self.counters.get(key, 0) + value

        event = MetricEvent(
            metric_name=name,
            metric_value=float(value),
            tags=tags or {},
            metadata={"type": "counter"}
        )
        self.metrics.append(event)

    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric (point-in-time value)"""
        key = self._make_key(name, tags)
        self.gauges[key] = value

        event = MetricEvent(
            metric_name=name,
            metric_value=value,
            tags=tags or {},
            metadata={"type": "gauge"}
        )
        self.metrics.append(event)

    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric (distribution of values)"""
        key = self._make_key(name, tags)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)

        event = MetricEvent(
            metric_name=name,
            metric_value=value,
            tags=tags or {},
            metadata={"type": "histogram"}
        )
        self.metrics.append(event)

    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create unique key for metric"""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value"""
        key = self._make_key(name, tags)
        return self.counters.get(key, 0)

    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current gauge value"""
        key = self._make_key(name, tags)
        return self.gauges.get(key)

    def get_histogram_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics"""
        key = self._make_key(name, tags)
        values = self.histograms.get(key, [])

        if not values:
            return {"count": 0, "min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}

        sorted_values = sorted(values)
        count = len(sorted_values)

        return {
            "count": count,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)]
        }

    def export_metrics(self, filepath: str) -> None:
        """Export metrics to file"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "counters": self.counters,
            "gauges": self.gauges,
            "histograms": {
                key: self.get_histogram_stats("", None)
                for key in self.histograms.keys()
            },
            "raw_events": [m.to_dict() for m in self.metrics[-1000:]]  # Last 1000 events
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.last_export_time = time.time()

    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()


class OpenTelemetryTracer:
    """
    OpenTelemetry distributed tracing

    Provides:
    - Trace context propagation
    - Span creation for operations
    - Trace export to backends (Jaeger, Zipkin, etc.)
    """

    def __init__(
        self,
        service_name: str = "deepagent",
        export_endpoint: Optional[str] = None
    ):
        self.service_name = service_name
        self.export_endpoint = export_endpoint
        self.otel_available = False
        self.tracer = None

        # Try to import OpenTelemetry
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
            from opentelemetry.sdk.resources import Resource

            # Create resource
            resource = Resource(attributes={
                "service.name": service_name
            })

            # Set up tracer provider
            provider = TracerProvider(resource=resource)

            # Add console exporter for development
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))

            # If export endpoint provided, add OTLP exporter
            if export_endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                    otlp_exporter = OTLPSpanExporter(endpoint=export_endpoint)
                    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                    print(f"OpenTelemetry exporting to {export_endpoint}")
                except ImportError:
                    print("Warning: OTLP exporter not available. Install with: pip install opentelemetry-exporter-otlp")

            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(__name__)
            self.otel_available = True
            print(f"OpenTelemetry tracing enabled for {service_name}")

        except ImportError:
            print("Warning: OpenTelemetry not installed. Tracing disabled.")
            print("Install with: pip install opentelemetry-api opentelemetry-sdk")

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Start a new trace span"""
        if self.otel_available and self.tracer:
            span = self.tracer.start_span(name)
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            return span
        else:
            # Return no-op context manager
            return NoOpSpan()

    def trace_operation(self, operation_name: str):
        """Decorator to trace an operation"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.start_span(operation_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


class NoOpSpan:
    """No-op span for when OpenTelemetry is unavailable"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key, value):
        pass


class ObservabilityManager:
    """
    Unified observability manager

    Integrates logging, metrics, and tracing
    """

    def __init__(
        self,
        service_name: str = "deepagent",
        log_level: str = "INFO",
        enable_metrics: bool = True,
        enable_tracing: bool = True,
        export_endpoint: Optional[str] = None
    ):
        self.service_name = service_name

        # Initialize components
        self.logger = StructuredLogger(name=service_name, log_level=log_level)

        self.metrics = MetricsCollector() if enable_metrics else None

        self.tracer = OpenTelemetryTracer(
            service_name=service_name,
            export_endpoint=export_endpoint
        ) if enable_tracing else None

    def log(self, level: str, event: str, **kwargs):
        """Log an event"""
        getattr(self.logger, level.lower())(event, **kwargs)

    def record_metric(self, metric_type: str, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric"""
        if not self.metrics:
            return

        if metric_type == "counter":
            self.metrics.record_counter(name, int(value), tags)
        elif metric_type == "gauge":
            self.metrics.record_gauge(name, value, tags)
        elif metric_type == "histogram":
            self.metrics.record_histogram(name, value, tags)

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Start a trace span"""
        if self.tracer:
            return self.tracer.start_span(name, attributes)
        return NoOpSpan()

    def export_metrics(self, filepath: str):
        """Export metrics to file"""
        if self.metrics:
            self.metrics.export_metrics(filepath)

    def get_stats(self) -> Dict[str, Any]:
        """Get observability statistics"""
        stats = {
            "service_name": self.service_name,
            "logger_available": self.logger.structlog_available,
            "metrics_available": self.metrics is not None,
            "tracing_available": self.tracer.otel_available if self.tracer else False
        }

        if self.metrics:
            stats["total_metrics"] = len(self.metrics.metrics)
            stats["counters"] = len(self.metrics.counters)
            stats["gauges"] = len(self.metrics.gauges)
            stats["histograms"] = len(self.metrics.histograms)

        return stats


# Convenience function
def create_observability(
    service_name: str = "deepagent",
    log_level: str = "INFO",
    export_endpoint: Optional[str] = None
) -> ObservabilityManager:
    """
    Create observability manager with default configuration

    Args:
        service_name: Name of the service
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        export_endpoint: Optional OTLP export endpoint

    Returns:
        Configured ObservabilityManager

    Example:
        >>> obs = create_observability("deepagent", "INFO")
        >>> obs.log("info", "agent_started", agent_id="123")
        >>> obs.record_metric("counter", "tool_executions", 1, {"tool": "search_pubmed"})
        >>> with obs.start_span("tool_execution"):
        ...     # perform operation
    """
    return ObservabilityManager(
        service_name=service_name,
        log_level=log_level,
        enable_metrics=True,
        enable_tracing=True,
        export_endpoint=export_endpoint
    )
