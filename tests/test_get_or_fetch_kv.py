import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from embcache import (
    KVCache, CacheConfig, EmbeddingFingerprint, KVFingerprint, LLMConfig, MetricsCollector,
)


def _emb_fp():
    return EmbeddingFingerprint(
        model_id="m1", embedding_dim=128, tokenizer_hash="t",
        chunking_strategy_hash="c", normalization_version="n",
        prompt_template_hash="p", dataset_version="d",
    )


def _kv_fp():
    return KVFingerprint(
        model_id="m1", llm_endpoint_hash="e",
        prompt_template_hash="p", dataset_version="d",
    )


def _make_cache():
    os.environ.setdefault("KV_TEST_KEY", "x")
    with patch("embcache._get_or_fetch_kv.CPUCache") as m_cpu, \
         patch("embcache._get_or_fetch_kv.GCSBackend") as m_gcs, \
         patch("embcache._get_or_fetch_kv.GDSBackend") as m_gds, \
         patch("embcache._get_or_fetch_kv.LLMClient") as m_llm:

        m_cpu.return_value.get_kv.return_value = None
        gcs_inst = m_gcs.return_value
        gcs_inst.get_kv = AsyncMock(return_value=None)
        gcs_inst.put_kv = AsyncMock(return_value=None)
        gds_inst = m_gds.return_value
        gds_inst.get_kv = AsyncMock(return_value=None)

        llm_inst = m_llm.return_value
        llm_inst.generate_kv_state = AsyncMock(return_value=b"GENERATED")
        llm_inst.close = AsyncMock(return_value=None)

        config = CacheConfig(
            embedding_fingerprint=_emb_fp(),
            kv_fingerprint=_kv_fp(),
            llm=LLMConfig(endpoint="http://x", api_key_env_var="KV_TEST_KEY", model_id="m"),
            gcs_bucket="",
        )
        metrics = MetricsCollector(namespace="test_kv")
        cache = KVCache(config, metrics)
        cache._gpu = None
        return cache, {
            "cpu": m_cpu.return_value,
            "gcs": gcs_inst,
            "gds": gds_inst,
            "llm": llm_inst,
        }


@pytest.mark.asyncio
async def test_cpu_hit():
    cache, mocks = _make_cache()
    mocks["cpu"].get_kv.return_value = b"cached"

    res = await cache.get_or_fetch_kv("doc")
    assert res.hit is True
    assert res.tier == "cpu_l2"
    assert res.kv_state == b"cached"
    mocks["llm"].generate_kv_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_cold_hit():
    cache, mocks = _make_cache()
    mocks["gcs"].get_kv = AsyncMock(return_value=b"from-gcs")

    res = await cache.get_or_fetch_kv("doc")
    assert res.hit is True
    assert res.tier == "cold"
    assert res.kv_state == b"from-gcs"
    mocks["llm"].generate_kv_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_full_miss_generates_via_llm():
    cache, mocks = _make_cache()

    res = await cache.get_or_fetch_kv("new doc")
    await asyncio.sleep(0)

    assert res.hit is False
    assert res.tier == "fetch"
    assert res.kv_state == b"GENERATED"
    mocks["cpu"].put_kv.assert_called()
    mocks["gcs"].put_kv.assert_awaited()


@pytest.mark.asyncio
async def test_concurrent_kv_requests_coalesce():
    cache, mocks = _make_cache()
    call_count = 0

    async def slow_gen(_doc):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        return b"VAL"

    mocks["llm"].generate_kv_state = AsyncMock(side_effect=slow_gen)

    t1 = asyncio.create_task(cache.get_or_fetch_kv("same"))
    await asyncio.sleep(0)
    t2 = asyncio.create_task(cache.get_or_fetch_kv("same"))

    r1, r2 = await asyncio.gather(t1, t2)
    assert r1.kv_state == r2.kv_state == b"VAL"
    assert call_count == 1
