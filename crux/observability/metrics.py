import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from functools import wraps
import threading
import json

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    from prometheus_client.core import CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from flask import Flask, jsonify, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


class ExporterType(Enum):
    CONSOLE = "console"
    OTLP = "otlp"
    DATADOG = "datadog"
    HONEYCOMB = "honeycomb"


@dataclass
class TelemetryConfig:
    enabled: bool = False
    service_name: str = "crux"
    service_version: str = "1.0.0"
    environment: str = "production"
    
    trace_enabled: bool = True
    trace_exporter: ExporterType = ExporterType.OTLP
    trace_endpoint: str = "http://localhost:4317"
    trace_insecure: bool = True
    trace_export_timeout: int = 30
    trace_max_export_batch_size: int = 512
    
    metrics_enabled: bool = True
    metrics_exporter: ExporterType = ExporterType.OTLP
    metrics_endpoint: str = "http://localhost:4317"
    metrics_export_interval: int = 10
    
    logging_enabled: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    
    dashboard_enabled: bool = True
    dashboard_port: int = 8080
    
    datadog_api_key: Optional[str] = None
    datadog_site: str = "datadoghq.com"
    honeycomb_api_key: Optional[str] = None
    honeycomb_dataset: str = "crux"
    
    correlation_header: str = "x-correlation-id"


class StructuredLogger:
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.logger = logging.getLogger("crux.observability")
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            if config.log_format == "json":
                handler.setFormatter(JsonFormatter())
            else:
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            self.logger.addHandler(handler)
        
        self._correlation_context: Dict[str, str] = {}
        self._lock = threading.Lock()
    
    def set_correlation_id(self, correlation_id: Optional[str] = None) -> str:
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        with self._lock:
            self._correlation_context["correlation_id"] = correlation_id
        return correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        with self._lock:
            return self._correlation_context.get("correlation_id")
    
    def _build_log_record(self, level: str, message: str, extra: Optional[Dict] = None) -> Dict:
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "logger": "crux.observability",
            "message": message,
            "service_name": self.config.service_name,
            "service_version": self.config.service_version,
            "environment": self.config.environment,
            "correlation_id": self.get_correlation_id(),
            "trace_id": self._get_trace_id(),
            "span_id": self._get_span_id(),
        }
        if extra:
            record["extra"] = extra
        return record
    
    def _get_trace_id(self) -> Optional[str]:
        if OPENTELEMETRY_AVAILABLE:
            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                return format(span.get_span_context().trace_id, '032x')
        return None
    
    def _get_span_id(self) -> Optional[str]:
        if OPENTELEMETRY_AVAILABLE:
            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                return format(span.get_span_context().span_id, '016x')
        return None
    
    def debug(self, message: str, extra: Optional[Dict] = None):
        if self.config.logging_enabled:
            record = self._build_log_record("DEBUG", message, extra)
            self.logger.debug(json.dumps(record))
    
    def info(self, message: str, extra: Optional[Dict] = None):
        if self.config.logging_enabled:
            record = self._build_log_record("INFO", message, extra)
            self.logger.info(json.dumps(record))
    
    def warning(self, message: str, extra: Optional[Dict] = None):
        if self.config.logging_enabled:
            record = self._build_log_record("WARNING", message, extra)
            self.logger.warning(json.dumps(record))
    
    def error(self, message: str, extra: Optional[Dict] = None):
        if self.config.logging_enabled:
            record = self._build_log_record("ERROR", message, extra)
            self.logger.error(json.dumps(record))
    
    def critical(self, message: str, extra: Optional[Dict] = None):
        if self.config.logging_enabled:
            record = self._build_log_record("CRITICAL", message, extra)
            self.logger.critical(json.dumps(record))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


class PrometheusMetricsCollector:
    def __init__(self, config: TelemetryConfig, registry: Optional["CollectorRegistry"] = None):
        self.config = config
        self.registry = registry
        
        if PROMETHEUS_AVAILABLE:
            self._setup_metrics()
        else:
            self._metrics = {}
    
    def _setup_metrics(self):
        reg = self.registry
        
        self._metrics = {
            "cache_hits": Counter(
                "crux_cache_hits_total",
                "Total number of cache hits",
                ["agent_id", "operation"],
                registry=reg
            ),
            "cache_misses": Counter(
                "crux_cache_misses_total",
                "Total number of cache misses",
                ["agent_id", "operation"],
                registry=reg
            ),
            "retrieval_latency": Histogram(
                "crux_retrieval_latency_seconds",
                "Retrieval operation latency in seconds",
                ["agent_id", "operation", "status"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                registry=reg
            ),
            "memory_count": Gauge(
                "crux_memory_count",
                "Number of memories stored per agent",
                ["agent_id", "memory_type"],
                registry=reg
            ),
            "memory_operations": Counter(
                "crux_memory_operations_total",
                "Total memory operations by type",
                ["agent_id", "operation", "status"],
                registry=reg
            ),
            "memory_size_bytes": Gauge(
                "crux_memory_size_bytes",
                "Total memory size in bytes per agent",
                ["agent_id"],
                registry=reg
            ),
            "hot_memories": Gauge(
                "crux_hot_memories_count",
                "Number of frequently accessed memories",
                ["agent_id"],
                registry=reg
            ),
            "embedding_latency": Histogram(
                "crux_embedding_latency_seconds",
                "Embedding generation latency in seconds",
                ["model", "dimension"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                registry=reg
            ),
            "vector_search_latency": Histogram(
                "crux_vector_search_latency_seconds",
                "Vector search latency in seconds",
                ["agent_id", "index_type", "top_k"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
                registry=reg
            ),
            "active_agents": Gauge(
                "crux_active_agents",
                "Number of active agents",
                [],
                registry=reg
            ),
        }
    
    def record_cache_hit(self, agent_id: str, operation: str = "search"):
        if PROMETHEUS_AVAILABLE and "cache_hits" in self._metrics:
            self._metrics["cache_hits"].labels(agent_id=agent_id, operation=operation).inc()
    
    def record_cache_miss(self, agent_id: str, operation: str = "search"):
        if PROMETHEUS_AVAILABLE and "cache_misses" in self._metrics:
            self._metrics["cache_misses"].labels(agent_id=agent_id, operation=operation).inc()
    
    def record_retrieval_latency(self, agent_id: str, operation: str, status: str, latency: float):
        if PROMETHEUS_AVAILABLE and "retrieval_latency" in self._metrics:
            self._metrics["retrieval_latency"].labels(
                agent_id=agent_id, operation=operation, status=status
            ).observe(latency)
    
    def set_memory_count(self, agent_id: str, memory_type: str, count: int):
        if PROMETHEUS_AVAILABLE and "memory_count" in self._metrics:
            self._metrics["memory_count"].labels(agent_id=agent_id, memory_type=memory_type).set(count)
    
    def record_memory_operation(self, agent_id: str, operation: str, status: str):
        if PROMETHEUS_AVAILABLE and "memory_operations" in self._metrics:
            self._metrics["memory_operations"].labels(
                agent_id=agent_id, operation=operation, status=status
            ).inc()
    
    def set_memory_size(self, agent_id: str, size_bytes: int):
        if PROMETHEUS_AVAILABLE and "memory_size_bytes" in self._metrics:
            self._metrics["memory_size_bytes"].labels(agent_id=agent_id).set(size_bytes)
    
    def set_hot_memories_count(self, agent_id: str, count: int):
        if PROMETHEUS_AVAILABLE and "hot_memories" in self._metrics:
            self._metrics["hot_memories"].labels(agent_id=agent_id).set(count)
    
    def record_embedding_latency(self, model: str, dimension: int, latency: float):
        if PROMETHEUS_AVAILABLE and "embedding_latency" in self._metrics:
            self._metrics["embedding_latency"].labels(model=model, dimension=dimension).observe(latency)
    
    def record_vector_search_latency(self, agent_id: str, index_type: str, top_k: int, latency: float):
        if PROMETHEUS_AVAILABLE and "vector_search_latency" in self._metrics:
            self._metrics["vector_search_latency"].labels(
                agent_id=agent_id, index_type=index_type, top_k=top_k
            ).observe(latency)
    
    def set_active_agents(self, count: int):
        if PROMETHEUS_AVAILABLE and "active_agents" in self._metrics:
            self._metrics["active_agents"].set(count)
    
    def get_metrics(self) -> bytes:
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry)
        return b""


class OpenTelemetryTracingManager:
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.tracer: Optional["trace.Tracer"] = None
        self.meter: Optional["metrics.Meter"] = None
        self._initialized = False
        self._lock = threading.Lock()
        
        if OPENTELEMETRY_AVAILABLE:
            self._setup()
    
    def _setup(self):
        with self._lock:
            if self._initialized:
                return
            
            resource = Resource.create({
                SERVICE_NAME: self.config.service_name,
                "service.version": self.config.service_version,
                "deployment.environment": self.config.environment,
            })
            
            if self.config.trace_enabled:
                self._setup_tracing(resource)
            
            if self.config.metrics_enabled:
                self._setup_metrics(resource)
            
            self._initialized = True
    
    def _setup_tracing(self, resource: Resource):
        provider = TracerProvider(resource=resource)
        
        exporter = self._get_trace_exporter()
        if exporter:
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
        
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(
            self.config.service_name,
            self.config.service_version
        )
    
    def _setup_metrics(self, resource: Resource):
        exporter = self._get_metrics_exporter()
        if exporter:
            reader = PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=self.config.metrics_export_interval * 1000
            )
            provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(provider)
            self.meter = metrics.get_meter(self.config.service_name)
    
    def _get_trace_exporter(self):
        if self.config.trace_exporter == ExporterType.OTLP:
            return OTLPSpanExporter(
                endpoint=self.config.trace_endpoint,
                insecure=self.config.trace_insecure,
            )
        elif self.config.trace_exporter == ExporterType.DATADOG:
            try:
                from opentelemetry.exporter.datadog import DdSpanExporter
                return DdSpanExporter(
                    agent_url="http://localhost:8126",
                    service=self.config.service_name,
                )
            except ImportError:
                pass
        elif self.config.trace_exporter == ExporterType.HONEYCOMB:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                return OTLPSpanExporter(
                    endpoint="https://api.honeycomb.io:443",
                    headers={"x-honeycomb-team": self.config.honeycomb_api_key or ""},
                )
            except ImportError:
                pass
        elif self.config.trace_exporter == ExporterType.CONSOLE:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            return ConsoleSpanExporter()
        return None
    
    def _get_metrics_exporter(self):
        if self.config.metrics_exporter == ExporterType.OTLP:
            return OTLPMetricExporter(
                endpoint=self.config.metrics_endpoint,
                insecure=self.config.metrics_insecure if hasattr(self.config, 'metrics_insecure') else True,
            )
        elif self.config.metrics_exporter == ExporterType.CONSOLE:
            return ConsoleMetricExporter()
        return None
    
    def start_span(
        self,
        name: str,
        operation: str,
        agent_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        attributes: Optional[Dict] = None,
        parent_span: Optional["trace.Span"] = None
    ) -> "Optional[trace.Span]":
        if not self.config.trace_enabled or not self.tracer:
            return None
        
        span_attributes = {
            "crux.operation": operation,
            "crux.service": self.config.service_name,
        }
        
        if agent_id:
            span_attributes["crux.agent_id"] = agent_id
        if memory_id:
            span_attributes["crux.memory_id"] = memory_id
        
        if attributes:
            span_attributes.update(attributes)
        
        ctx = trace.set_span_in_context(parent_span) if parent_span else None
        span = self.tracer.start_span(name, attributes=span_attributes, context=ctx)
        
        return span
    
    def end_span(self, span: Optional["trace.Span"], status: str = "OK", error_message: Optional[str] = None):
        if span:
            if status == "ERROR":
                span.set_status(Status(StatusCode.ERROR, error_message or "Unknown error"))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()


class TelemetryContext:
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self._lock = threading.local()
    
    def __enter__(self):
        if not hasattr(self._lock, 'context'):
            self._lock.context = {}
        return self._lock.context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def set(self, key: str, value: Any):
        if hasattr(self._lock, 'context'):
            self._lock.context[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self._lock, 'context'):
            return self._lock.context.get(key, default)
        return default


class MetricsCollector:
    def