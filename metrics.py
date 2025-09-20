"""
Prometheus metrics collection for Terra Constellata.
Provides comprehensive monitoring metrics for the application.
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
import time
from typing import Optional
from .logging_config import log_metrics

# Application Info
app_info = Info("terra_constellata_info", "Application information")
app_info.info({"version": "1.0.0", "service": "backend"})

# Request Metrics
REQUEST_COUNT = Counter(
    "terra_constellata_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "terra_constellata_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# Business Metrics
WORKFLOW_STARTED = Counter(
    "terra_constellata_workflows_started_total",
    "Total number of workflows started",
    ["workflow_type"],
)

WORKFLOW_COMPLETED = Counter(
    "terra_constellata_workflows_completed_total",
    "Total number of workflows completed",
    ["workflow_type", "status"],
)

ARTWORK_GENERATED = Counter(
    "terra_constellata_artworks_generated_total",
    "Total number of artworks generated",
    ["agent_type", "style"],
)

KNOWLEDGE_ENTRIES = Counter(
    "terra_constellata_knowledge_entries_total",
    "Total number of knowledge entries created",
    ["entry_type"],
)

# System Metrics
ACTIVE_CONNECTIONS = Gauge(
    "terra_constellata_active_connections", "Number of active connections"
)

MEMORY_USAGE = Gauge("terra_constellata_memory_usage_bytes", "Memory usage in bytes")

CPU_USAGE = Gauge("terra_constellata_cpu_usage_percent", "CPU usage percentage")

# Error Metrics
ERROR_COUNT = Counter(
    "terra_constellata_errors_total",
    "Total number of errors",
    ["error_type", "endpoint"],
)

# Agent Metrics
AGENT_REQUESTS = Counter(
    "terra_constellata_agent_requests_total",
    "Total number of agent requests",
    ["agent_type", "operation"],
)

AGENT_LATENCY = Histogram(
    "terra_constellata_agent_duration_seconds",
    "Agent operation duration in seconds",
    ["agent_type", "operation"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# Database Metrics
DB_CONNECTIONS = Gauge(
    "terra_constellata_db_connections_active",
    "Number of active database connections",
    ["db_type"],
)

DB_QUERY_LATENCY = Histogram(
    "terra_constellata_db_query_duration_seconds",
    "Database query duration in seconds",
    ["db_type", "operation"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
)

# Learning Metrics
FEEDBACK_RECEIVED = Counter(
    "terra_constellata_feedback_received_total",
    "Total number of feedback entries received",
    ["feedback_type", "rating"],
)

MODEL_TRAINING_TIME = Histogram(
    "terra_constellata_model_training_duration_seconds",
    "Model training duration in seconds",
    ["model_type"],
    buckets=[60, 300, 600, 1800, 3600, 7200],
)


# Utility functions
def record_request(method: str, endpoint: str, status_code: int, duration: float):
    """Record HTTP request metrics."""
    REQUEST_COUNT.labels(
        method=method, endpoint=endpoint, status_code=status_code
    ).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

    # Log to structured logging as well
    log_metrics(
        metric_name="request_duration",
        metric_value=duration,
        tags={"method": method, "endpoint": endpoint, "status_code": status_code},
    )


def record_workflow_start(workflow_type: str):
    """Record workflow start."""
    WORKFLOW_STARTED.labels(workflow_type=workflow_type).inc()


def record_workflow_completion(workflow_type: str, status: str):
    """Record workflow completion."""
    WORKFLOW_COMPLETED.labels(workflow_type=workflow_type, status=status).inc()


def record_artwork_generation(agent_type: str, style: str):
    """Record artwork generation."""
    ARTWORK_GENERATED.labels(agent_type=agent_type, style=style).inc()


def record_knowledge_entry(entry_type: str):
    """Record knowledge entry creation."""
    KNOWLEDGE_ENTRIES.labels(entry_type=entry_type).inc()


def record_error(error_type: str, endpoint: Optional[str] = None):
    """Record error occurrence."""
    ERROR_COUNT.labels(error_type=error_type, endpoint=endpoint or "unknown").inc()


def record_agent_request(
    agent_type: str, operation: str, duration: Optional[float] = None
):
    """Record agent request."""
    AGENT_REQUESTS.labels(agent_type=agent_type, operation=operation).inc()
    if duration:
        AGENT_LATENCY.labels(agent_type=agent_type, operation=operation).observe(
            duration
        )


def record_db_operation(db_type: str, operation: str, duration: float):
    """Record database operation."""
    DB_QUERY_LATENCY.labels(db_type=db_type, operation=operation).observe(duration)


def record_feedback(feedback_type: str, rating: int):
    """Record user feedback."""
    FEEDBACK_RECEIVED.labels(feedback_type=feedback_type, rating=str(rating)).inc()


def update_system_metrics():
    """Update system-level metrics."""
    import psutil

    # Memory usage
    memory = psutil.virtual_memory()
    MEMORY_USAGE.set(memory.used)

    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    CPU_USAGE.set(cpu_percent)

    # Log system metrics
    log_metrics("memory_usage_percent", memory.percent)
    log_metrics("cpu_usage_percent", cpu_percent)


def get_metrics():
    """Get all metrics in Prometheus format."""
    return generate_latest()


# Initialize system metrics on import
update_system_metrics()
