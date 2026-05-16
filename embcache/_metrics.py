import json
import logging
import time
from datetime import datetime, timezone
import sys
from typing import Literal

from prometheus_client import (
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

# --- Prometheus Metrics Registration ---

METRICS = {}

def _register_metric(cls, name, documentation, labels=None, **kwargs):
    if name in METRICS:
        return METRICS[name]
    try:
        metric = cls(name, documentation, labelnames=labels or ["namespace"], **kwargs)
        METRICS[name] = metric
        return metric
    except ValueError:
        # Already registered in global REGISTRY (e.g. duplicate import); reuse existing collector.
        existing = getattr(REGISTRY, "_names_to_collectors", {}).get(name)
        if existing is not None:
            METRICS[name] = existing
            return existing
        raise

# Counters
exact_hits = _register_metric(Counter, "embcache_exact_hits_total", "Total exact cache hits")
semantic_hits = _register_metric(Counter, "embcache_semantic_hits_total", "Total semantic cache hits")
gpu_l1_hits = _register_metric(Counter, "embcache_gpu_l1_hits_total", "Total GPU L1 hits")
cpu_l2_hits = _register_metric(Counter, "embcache_cpu_l2_hits_total", "Total CPU L2 hits")
cold_store_hits = _register_metric(Counter, "embcache_cold_store_hits_total", "Total cold store hits")
kv_hits = _register_metric(Counter, "embcache_kv_hits_total", "Total KV hits")
kv_misses = _register_metric(Counter, "embcache_kv_misses_total", "Total KV misses")
evictions = _register_metric(Counter, "embcache_evictions_total", "Total evictions", labels=["namespace", "entry_type"])
gcs_write_failures = _register_metric(Counter, "embcache_gcs_write_failures_total", "Total GCS write failures")
gcs_read_failures = _register_metric(Counter, "embcache_gcs_read_failures_total", "Total GCS read failures")
fingerprint_mismatches = _register_metric(Counter, "embcache_fingerprint_mismatches_total", "Total fingerprint mismatches")
faiss_writes_dropped = _register_metric(Counter, "embcache_faiss_writes_dropped_total", "Total FAISS writes dropped")

# Gauges
inflight_coalesced = _register_metric(Gauge, "embcache_inflight_coalesced_requests", "In-flight coalesced requests")
faiss_queue_depth = _register_metric(Gauge, "embcache_faiss_write_queue_depth", "FAISS write queue depth")
slab_utilization = _register_metric(Gauge, "embcache_slab_utilization_percent", "Slab utilization percent")
slab_embedding_bytes = _register_metric(Gauge, "embcache_slab_embedding_bytes", "Slab embedding bytes")
slab_kv_bytes = _register_metric(Gauge, "embcache_slab_kv_bytes", "Slab KV bytes")
gpu_mem_reserved = _register_metric(Gauge, "embcache_gpu_memory_reserved_bytes", "GPU memory reserved bytes")
gpu_mem_allocated = _register_metric(Gauge, "embcache_gpu_memory_allocated_bytes", "GPU memory allocated bytes")
namespace_count = _register_metric(Gauge, "embcache_cache_key_namespace_count", "Cache key namespace count")

# Histograms
kv_gen_latency = _register_metric(Histogram, "embcache_kv_generation_latency_seconds", "KV generation latency",
                                  buckets=(.1, .5, 1, 2, 5, 10, 30))
kv_state_size = _register_metric(Histogram, "embcache_kv_state_size_bytes", "KV state size in bytes",
                                 buckets=(1e4, 1e5, 1e6, 1e7, 1e8, 1e9))
gds_transfer_latency = _register_metric(Histogram, "embcache_gds_transfer_latency_seconds", "GDS transfer latency",
                                        buckets=(.001, .005, .01, .05, .1, .5, 1))
h2d_transfer_latency = _register_metric(Histogram, "embcache_h2d_transfer_latency_seconds", "H2D transfer latency",
                                        buckets=(.001, .005, .01, .05, .1, .5, 1))

# --- MetricsCollector Class ---

class MetricsCollector:
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace

    def record_exact_hit(self): exact_hits.labels(self.namespace).inc()
    def record_semantic_hit(self): semantic_hits.labels(self.namespace).inc()
    def record_gpu_l1_hit(self): gpu_l1_hits.labels(self.namespace).inc()
    def record_cpu_l2_hit(self): cpu_l2_hits.labels(self.namespace).inc()
    def record_cold_store_hit(self): cold_store_hits.labels(self.namespace).inc()
    def record_kv_hit(self): kv_hits.labels(self.namespace).inc()
    def record_kv_miss(self): kv_misses.labels(self.namespace).inc()
    def record_eviction(self, entry_type: Literal["embedding", "kv"]):
        evictions.labels(self.namespace, entry_type).inc()
    def record_gcs_write_failure(self): gcs_write_failures.labels(self.namespace).inc()
    def record_gcs_read_failure(self): gcs_read_failures.labels(self.namespace).inc()
    def record_fingerprint_mismatch(self): fingerprint_mismatches.labels(self.namespace).inc()
    def record_faiss_write_dropped(self): faiss_writes_dropped.labels(self.namespace).inc()

    def set_inflight(self, n: int): inflight_coalesced.labels(self.namespace).set(n)
    def set_faiss_queue_depth(self, n: int): faiss_queue_depth.labels(self.namespace).set(n)
    def set_slab_utilization(self, pct: float): slab_utilization.labels(self.namespace).set(pct)
    def set_slab_bytes(self, embedding_bytes: int, kv_bytes: int):
        slab_embedding_bytes.labels(self.namespace).set(embedding_bytes)
        slab_kv_bytes.labels(self.namespace).set(kv_bytes)
    def set_gpu_memory(self, reserved: int, allocated: int):
        gpu_mem_reserved.labels(self.namespace).set(reserved)
        gpu_mem_allocated.labels(self.namespace).set(allocated)
    def set_namespace_count(self, n: int): namespace_count.labels(self.namespace).set(n)

    def observe_kv_generation(self, seconds: float): kv_gen_latency.labels(self.namespace).observe(seconds)
    def observe_kv_state_size(self, bytes_: int): kv_state_size.labels(self.namespace).observe(bytes_)
    def observe_gds_transfer(self, seconds: float): gds_transfer_latency.labels(self.namespace).observe(seconds)
    def observe_h2d_transfer(self, seconds: float): h2d_transfer_latency.labels(self.namespace).observe(seconds)

# --- Structured JSON Logger ---

class JSONFormatter(logging.Formatter):
    _RESERVED = {
        'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
        'funcName', 'levelname', 'levelno', 'lineno', 'module',
        'msecs', 'msg', 'name', 'pathname', 'process', 'processName',
        'relativeCreated', 'stack_info', 'thread', 'threadName', 'message',
    }

    def format(self, record):
        log_entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec='milliseconds').replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = {k: v for k, v in record.__dict__.items() if k not in self._RESERVED}
        log_entry.update(extra)
        return json.dumps(log_entry, default=str)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger

_log = get_logger(__name__)

def start_metrics_server(port: int = 9090) -> None:
    try:
        start_http_server(port)
        _log.info(f"Metrics server started on port {port}")
    except Exception as e:
        _log.error(f"Failed to start metrics server: {e}")
