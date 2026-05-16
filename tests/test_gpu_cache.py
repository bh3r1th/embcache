import numpy as np
import pytest
from unittest.mock import MagicMock

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

from embcache._metrics import MetricsCollector

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA required for GPUCache")


def _metrics():
    return MagicMock(spec=MetricsCollector)


def _make_cache(embedding_dim=8, kv_slot_size=1024, fraction=0.001, embedding_fraction=0.5):
    from embcache._gpu_cache import GPUCache
    return GPUCache(
        embedding_dim=embedding_dim,
        kv_slot_size=kv_slot_size,
        gpu_cache_max_fraction=fraction,
        embedding_fraction=embedding_fraction,
        metrics=_metrics(),
    )


def test_embedding_put_get_roundtrip_ndarray():
    cache = _make_cache()
    vec = np.arange(8, dtype=np.float32)
    cache.put_embedding("k1", vec)
    got = cache.get_embedding("k1")
    assert got is not None
    np.testing.assert_array_almost_equal(np.asarray(got, dtype=np.float32), vec)


def test_embedding_put_get_roundtrip_list():
    cache = _make_cache()
    vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    cache.put_embedding("k1", vec)
    got = cache.get_embedding("k1")
    assert got is not None
    np.testing.assert_array_almost_equal(
        np.asarray(got, dtype=np.float32),
        np.asarray(vec, dtype=np.float32),
    )


def test_kv_returns_exact_bytes_not_padded():
    cache = _make_cache(kv_slot_size=1024)
    state = b"hello world"
    cache.put_kv("k1", state)
    got = cache.get_kv("k1")
    assert got == state


def test_invalidate():
    cache = _make_cache()
    cache.put_embedding("k", np.zeros(8, dtype=np.float32))
    assert cache.invalidate("k") is True
    assert cache.get_embedding("k") is None
    assert cache.invalidate("missing") is False


def test_embedding_fraction_zero_allows_kv_only():
    cache = _make_cache(embedding_fraction=0.0)
    assert cache.n_embedding_slots == 0
    assert cache.n_kv_slots > 0
    cache.put_kv("k", b"abc")
    assert cache.get_kv("k") == b"abc"


def test_embedding_fraction_one_allows_emb_only():
    cache = _make_cache(embedding_fraction=1.0)
    assert cache.n_kv_slots == 0
    cache.put_embedding("k", np.zeros(8, dtype=np.float32))
    assert cache.get_embedding("k") is not None
