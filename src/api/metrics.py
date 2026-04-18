"""
Prometheus metrics for the Procurement Digital Twin API.

Exposes a /metrics endpoint (standard Prometheus text format).
Counters and histograms are updated by route handlers via the helpers below.

Metrics exported
----------------
  procurement_api_requests_total{endpoint, status}     Counter
  procurement_api_duration_seconds{endpoint}           Histogram
  procurement_api_cache_hits_total                     Counter
  procurement_api_cache_misses_total                   Counter
  procurement_api_active_simulations                   Gauge
  procurement_api_batch_size{endpoint}                 Histogram
  procurement_api_errors_total{endpoint, error_type}   Counter
"""

import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter, Histogram, Gauge,
        generate_latest, CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False
    logger.warning("prometheus-client not installed — /metrics endpoint will return 503.")


if _PROM_AVAILABLE:
    requests_total = Counter(
        "procurement_api_requests_total",
        "Total API requests by endpoint and HTTP status",
        ["endpoint", "status"],
    )
    request_duration = Histogram(
        "procurement_api_duration_seconds",
        "API request wall-clock duration in seconds",
        ["endpoint"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    cache_hits_total = Counter(
        "procurement_api_cache_hits_total",
        "Number of simulation cache hits",
    )
    cache_misses_total = Counter(
        "procurement_api_cache_misses_total",
        "Number of simulation cache misses",
    )
    active_simulations = Gauge(
        "procurement_api_active_simulations",
        "Number of simulation requests currently in flight",
    )
    batch_size_hist = Histogram(
        "procurement_api_batch_size",
        "Number of procedures per /batch request",
        buckets=(1, 2, 5, 10, 15, 20),
    )
    errors_total = Counter(
        "procurement_api_errors_total",
        "Simulation errors by endpoint and error type",
        ["endpoint", "error_type"],
    )
else:
    # Stub objects so callers don't need to guard every call
    class _Stub:
        def inc(self, *a, **k): pass
        def dec(self, *a, **k): pass
        def observe(self, *a, **k): pass
        def labels(self, *a, **k): return self
        def time(self): return self
        def __enter__(self): return self
        def __exit__(self, *a): pass

    requests_total = cache_hits_total = cache_misses_total = _Stub()
    request_duration = active_simulations = batch_size_hist = errors_total = _Stub()


def prometheus_response():
    """Return (body_bytes, content_type) for the /metrics endpoint."""
    if not _PROM_AVAILABLE:
        return None, None
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


@contextmanager
def track_simulation(endpoint: str):
    """Context manager: increment active gauge, record duration and status."""
    active_simulations.inc()
    t0 = time.perf_counter()
    status = "2xx"
    try:
        yield
    except Exception:
        status = "5xx"
        errors_total.labels(endpoint=endpoint, error_type="exception").inc()
        raise
    finally:
        active_simulations.dec()
        elapsed = time.perf_counter() - t0
        request_duration.labels(endpoint=endpoint).observe(elapsed)
        requests_total.labels(endpoint=endpoint, status=status).inc()


def record_cache_hit():
    cache_hits_total.inc()


def record_cache_miss():
    cache_misses_total.inc()


def record_request(endpoint: str, status: str):
    requests_total.labels(endpoint=endpoint, status=status).inc()
