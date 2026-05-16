import numpy as np
import pytest
from unittest.mock import MagicMock

from embcache._cpu_cache import CPUCache
from embcache._metrics import MetricsCollector


def _metrics():
    return MagicMock(spec=MetricsCollector)


def test_put_get_embedding_roundtrip():
    cache = CPUCache(max_embedding_bytes=1024, max_kv_bytes=0, metrics=_metrics())
    vec = np.arange(10, dtype=np.float32)
    cache.put_embedding("k1", vec)
    got = cache.get_embedding("k1")
    assert got is not None
    np.testing.assert_array_equal(got, vec)


def test_put_get_kv_roundtrip():
    cache = CPUCache(max_embedding_bytes=0, max_kv_bytes=1024, metrics=_metrics())
    cache.put_kv("k1", b"hello world")
    assert cache.get_kv("k1") == b"hello world"


def test_str_kv_is_encoded():
    cache = CPUCache(max_embedding_bytes=0, max_kv_bytes=1024, metrics=_metrics())
    cache.put_kv("k1", "hello")
    assert cache.get_kv("k1") == b"hello"


def test_oversized_embedding_rejected():
    cache = CPUCache(max_embedding_bytes=100, max_kv_bytes=0, metrics=_metrics())
    vec = np.zeros(1000, dtype=np.float32)  # 4000 bytes > 100
    cache.put_embedding("k", vec)
    assert cache.get_embedding("k") is None


def test_lru_eviction_embedding():
    m = _metrics()
    cache = CPUCache(max_embedding_bytes=100, max_kv_bytes=0, metrics=m)
    small = np.zeros(10, dtype=np.float32)  # 40 bytes
    cache.put_embedding("k1", small)
    cache.put_embedding("k2", small)
    cache.put_embedding("k3", small)  # 80+40=120 > 100 → evict k1

    assert cache.get_embedding("k1") is None
    assert cache.get_embedding("k2") is not None
    assert cache.get_embedding("k3") is not None
    m.record_eviction.assert_called_with("embedding")


def test_lru_eviction_kv():
    m = _metrics()
    cache = CPUCache(max_embedding_bytes=0, max_kv_bytes=25, metrics=m)
    cache.put_kv("k1", b"x" * 10)
    cache.put_kv("k2", b"x" * 10)
    cache.put_kv("k3", b"x" * 10)  # 20+10=30 > 25 → evict k1
    assert cache.get_kv("k1") is None
    assert cache.get_kv("k2") == b"x" * 10
    m.record_eviction.assert_called_with("kv")


def test_get_promotes_to_mru():
    cache = CPUCache(max_embedding_bytes=100, max_kv_bytes=0, metrics=_metrics())
    small = np.zeros(10, dtype=np.float32)
    cache.put_embedding("k1", small)
    cache.put_embedding("k2", small)
    cache.put_embedding("k3", small)  # evicts k1 (oldest)

    # Touch k2 to promote
    cache.get_embedding("k2")
    cache.put_embedding("k4", small)  # should evict k3 now, not k2

    assert cache.get_embedding("k2") is not None
    assert cache.get_embedding("k3") is None
    assert cache.get_embedding("k4") is not None


def test_invalidate():
    cache = CPUCache(max_embedding_bytes=1024, max_kv_bytes=1024, metrics=_metrics())
    vec = np.zeros(8, dtype=np.float32)
    cache.put_embedding("k", vec)
    cache.put_kv("k", b"data")

    assert cache.invalidate("k") is True
    assert cache.get_embedding("k") is None
    assert cache.get_kv("k") is None
    assert cache.invalidate("missing") is False


def test_disabled_pools_skip():
    cache = CPUCache(max_embedding_bytes=0, max_kv_bytes=0, metrics=_metrics())
    cache.put_embedding("k", np.zeros(4, dtype=np.float32))
    cache.put_kv("k", b"x")
    assert cache.get_embedding("k") is None
    assert cache.get_kv("k") is None


def test_stats_reports_state():
    cache = CPUCache(max_embedding_bytes=1024, max_kv_bytes=1024, metrics=_metrics())
    cache.put_embedding("a", np.zeros(8, dtype=np.float32))
    cache.put_kv("b", b"xxxx")
    s = cache.stats()
    assert s["embedding_count"] == 1
    assert s["kv_count"] == 1
    assert s["embedding_bytes"] == 32
    assert s["kv_bytes"] == 4
