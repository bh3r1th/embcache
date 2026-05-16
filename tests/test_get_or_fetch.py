import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from embcache import (
    EmbeddingCache, CacheConfig, EmbeddingFingerprint, FAISSIndexConfig, MetricsCollector,
)


def _fp():
    return EmbeddingFingerprint(
        model_id="m1",
        embedding_dim=128,
        tokenizer_hash="t",
        chunking_strategy_hash="c",
        normalization_version="n",
        prompt_template_hash="p",
        dataset_version="d",
    )


def _make_cache():
    """Build EmbeddingCache with all subcomponents patched to mocks."""
    with patch("embcache._get_or_fetch.CPUCache") as m_cpu, \
         patch("embcache._get_or_fetch.GCSBackend") as m_gcs, \
         patch("embcache._get_or_fetch.GDSBackend") as m_gds, \
         patch("embcache._get_or_fetch.ExactIndex") as m_exact, \
         patch("embcache._get_or_fetch.FAISSIndex") as m_faiss:

        m_cpu.return_value.get_embedding.return_value = None
        m_exact.return_value.get.return_value = None

        # GCS/GDS are async
        gcs_inst = m_gcs.return_value
        gcs_inst.get_embedding = AsyncMock(return_value=None)
        gcs_inst.put_embedding = AsyncMock(return_value=None)
        gds_inst = m_gds.return_value
        gds_inst.get_embedding = AsyncMock(return_value=None)

        faiss_inst = m_faiss.return_value
        faiss_inst.add = AsyncMock(return_value=None)
        faiss_inst.search = AsyncMock(return_value=[])
        faiss_inst.close = AsyncMock(return_value=None)

        config = CacheConfig(embedding_fingerprint=_fp(), faiss=FAISSIndexConfig(), gcs_bucket="")
        metrics = MetricsCollector(namespace="test")
        cache = EmbeddingCache(config, metrics)
        # Force CPU tier (skip GPU init)
        cache._gpu = None
        return cache, {
            "cpu": m_cpu.return_value,
            "gcs": gcs_inst,
            "gds": gds_inst,
            "exact": m_exact.return_value,
            "faiss": faiss_inst,
        }


@pytest.mark.asyncio
async def test_exact_hit():
    cache, mocks = _make_cache()
    vec = np.arange(128, dtype=np.float32)
    mocks["exact"].get.return_value = vec

    fetch_fn = AsyncMock(return_value=[0.0] * 128)
    res = await cache.get_or_fetch("hello", fetch_fn=fetch_fn)

    assert res.hit is True
    assert res.tier == "exact"
    assert res.embedding == vec.tolist()
    fetch_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_cpu_l2_hit_promotes_none_when_no_gpu():
    cache, mocks = _make_cache()
    vec = np.arange(128, dtype=np.float32)
    mocks["cpu"].get_embedding.return_value = vec

    fetch_fn = AsyncMock(return_value=[0.0] * 128)
    res = await cache.get_or_fetch("doc", fetch_fn=fetch_fn)

    assert res.hit is True
    assert res.tier == "cpu_l2"
    fetch_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_full_miss_calls_fetch_and_backfills():
    cache, mocks = _make_cache()
    vec_list = [float(i) for i in range(128)]
    fetch_fn = AsyncMock(return_value=vec_list)

    res = await cache.get_or_fetch("new", fetch_fn=fetch_fn)
    # Yield so the create_task'd gcs.put_embedding actually runs
    await asyncio.sleep(0)

    assert res.hit is False
    assert res.tier == "fetch"
    assert res.embedding == vec_list

    mocks["exact"].put.assert_called()
    mocks["cpu"].put_embedding.assert_called()
    mocks["faiss"].add.assert_awaited()
    mocks["gcs"].put_embedding.assert_awaited()


@pytest.mark.asyncio
async def test_cold_store_hit_marked_as_hit():
    cache, mocks = _make_cache()
    vec = np.ones(128, dtype=np.float32)
    mocks["gcs"].get_embedding = AsyncMock(return_value=vec)

    fetch_fn = AsyncMock(return_value=[0.0] * 128)
    res = await cache.get_or_fetch("cold-key", fetch_fn=fetch_fn)

    assert res.hit is True
    assert res.tier == "cold"
    fetch_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_semantic_hit_via_query_vector():
    cache, mocks = _make_cache()
    near = np.arange(128, dtype=np.float32)
    mocks["faiss"].search = AsyncMock(return_value=[("emb:near", 0.95)])
    # First lookup for matched_key in exact returns the stored vec
    mocks["exact"].get.side_effect = lambda k: near if k == "emb:near" else None

    fetch_fn = AsyncMock(return_value=[0.0] * 128)
    res = await cache.get_or_fetch(
        "semantic", fetch_fn=fetch_fn, query_vector=list(range(128))
    )

    assert res.hit is True
    assert res.tier == "semantic"
    fetch_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_requests_coalesce():
    cache, _mocks = _make_cache()
    call_count = 0

    async def slow_fetch(_text):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        return [1.0] * 128

    t1 = asyncio.create_task(cache.get_or_fetch("same", fetch_fn=slow_fetch))
    # Give t1 a chance to register the inflight future
    await asyncio.sleep(0)
    t2 = asyncio.create_task(cache.get_or_fetch("same", fetch_fn=slow_fetch))

    r1, r2 = await asyncio.gather(t1, t2)
    assert r1.embedding == r2.embedding
    assert call_count == 1
