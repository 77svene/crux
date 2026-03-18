import logging
import time
import uuid
import os
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Callable, Union
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.trace.sampling import Sampler, TraceIdRatioBased
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.trace.propagation.context import Context
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GRPCSpanExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter as GRPCMetricExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as HTTPMetricExporter
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = metrics = None
    TracerProvider = ConsoleSpanExporter = BatchSpanProcessor = None
    MeterProvider = PeriodicExportingMetricReader = None
    Resource = SERVICE_NAME = SERVICE_VERSION = None
    Status = StatusCode = None
    TraceContextTextMapPropagator = set_global_textmap = Context = None
    GRPCSpanExporter = HTTPSpanExporter = GRPCMetricExporter = HTTPMetricExporter = None

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = None

try:
    from flask import Flask, jsonify, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = jsonify = Response = None

try:
    from datadog import DogStatsd
    DATADOG_AVAILABLE = True
except ImportError:
    DATADOG_AVAILABLE = False
    DogStatsd = None


class ExporterType(Enum):
    CONSOLE = "console"
    OTLP_GRPC = "otlp_grpc"
    OTLP_HTTP = "otlp_http"
    DATADOG = "datadog"
    HONEYCOMB = "honeycomb"


@dataclass
class TelemetryConfig:
    enabled: bool = False
    service_name: str = "crux"
    service_version: str = "1.0.0"
    exporter_type: Union[ExporterType, str] = ExporterType.CONSOLE
    otlp_endpoint: Optional[str] = None
    datadog_api_key: Optional[str] = None
    honeycomb_api_key: Optional[str] = None
    honeycomb_dataset: Optional[str] = None
    log_level: str = "INFO"
    correlation_id_header: str = "X-Correlation-ID"
    trace_id_header: str = "X-Trace-ID"
    enable_prometheus_endpoint: bool = True
    prometheus_port: int = 9090
    dashboard_port: int = 5000
    export_interval_ms: int = 5000
    sample_rate: float = 1.0
    enable_flask_dashboard: bool = True
    
    def __post_init__(self):
        if isinstance(self.exporter_type, str):
            try:
                self.exporter_type = ExporterType(self.exporter_type)
            except ValueError:
                self.exporter_type = ExporterType.CONSOLE
        
        self.otlp_endpoint = self.otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        self.datadog_api_key = self.datadog_api_key or os.getenv("DATADOG_API_KEY")
        self.honeycomb_api_key = self.honeycomb_api_key or os.getenv("HONEYCOMB_API_KEY")
        self.honeycomb_dataset = self.honeycomb_dataset or os.getenv("HONEYCOMB_DATASET", "crux")
        
        if not OTEL_AVAILABLE and self.exporter_type not in [ExporterType.CONSOLE, ExporterType.DATADOG]:
            logger.warning(f"OpenTelemetry not available, falling back to {ExporterType.CONSOLE}")
            self.exporter_type = ExporterType.CONSOLE
    
    @classmethod
    def from_env(cls) -> "TelemetryConfig":
        env_enabled = os.getenv("MEM0_TELEMETRY_ENABLED", "false").lower() in ("true", "1", "yes")
        return cls(
            enabled=env_enabled,
            exporter_type=os.getenv("MEM0_EXPORTER_TYPE", "console"),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            service_name=os.getenv("MEM0_SERVICE_NAME", "crux"),
            service_version=os.getenv("MEM0_SERVICE_VERSION", "1.0.0"),
            log_level=os.getenv("MEM0_LOG_LEVEL", "INFO"),
            correlation_id_header=os.getenv("MEM0_CORRELATION_ID_HEADER", "X-Correlation-ID"),
            trace_id_header=os.getenv("MEM0_TRACE_ID_HEADER", "X-Trace-ID"),
            enable_prometheus_endpoint=os.getenv("MEM0_PROMETHEUS_ENABLED", "true").lower() in ("true", "1", "yes"),
            prometheus_port=int(os.getenv("MEM0_PROMETHEUS_PORT", "9090")),
            dashboard_port=int(os.getenv("MEM0_DASHBOARD_PORT", "5000")),
            export_interval_ms=int(os.getenv("MEM0_EXPORT_INTERVAL_MS", "5000")),
            sample_rate=float(os.getenv("MEM0_SAMPLE_RATE", "1.0")),
            enable_flask_dashboard=os.getenv("MEM0_DASHBOARD_ENABLED", "true").lower() in ("true", "1", "yes"),
        )


class MetricsCollector:
    def __init__(self, service_name: str, enable_prometheus: bool = True):
        self.service_name = service_name
        self.prometheus_available = PROMETHEUS_AVAILABLE and enable_prometheus
        self._lock = threading.Lock()
        
        self._metrics: Dict[str, Any] = {}
        self._counters: Dict[str, int] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._gauges: Dict[str, float] = {}
        self._hot_memories: Dict[str, int] = {}
        self._agent_activity: Dict[str, Dict[str, Any]] = {}
        
        if self.prometheus_available:
            self._init_prometheus_metrics()
        else:
            self._init_fallback_metrics()
    
    def _init_prometheus_metrics(self):
        self.cache_hits = Counter(
            "crux_cache_hits_total",
            "Total number of cache hits",
            ["agent_id"]
        )
        self.cache_misses = Counter(
            "crux_cache_misses_total",
            "Total number of cache misses",
            ["agent_id"]
        )
        self.retrieval_latency_seconds = Histogram(
            "crux_retrieval_latency_seconds",
            "Retrieval latency in seconds",
            ["agent_id", "operation"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        self.operation_latency_seconds = Histogram(
            "crux_operation_latency_seconds",
            "Memory operation latency in seconds",
            ["operation", "status"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0