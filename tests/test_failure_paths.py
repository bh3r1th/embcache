import asyncio
import pytest
import numpy as np
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock, patch

from embcache._gcs_backend import GCSBackend
from embcache._llm_client import LLMClient
from embcache._cpu_cache import CPUCache
from embcache._faiss_index import FAISSIndex
from embcache._fingerprint import EmbeddingFingerprint
from embcache._metrics import MetricsCollector
from embcache._config import LLMConfig, FAISSIndexConfig


@pytest.mark.asyncio
async def test_gcs_backend_read_failures():
    metrics = MagicMock(spec=MetricsCollector)
    with patch("embcache._gcs_backend.storage") as mock_storage:
        backend = GCSBackend("bucket", "prefix", metrics)

        # Simulated 404 (NotFound surfaces via getattr(e, 'code', None) == 404)
        err_404 = Exception("not found")
        err_404.code = 404

        mock_blob = MagicMock()
        mock_blob.download_as_bytes.side_effect = err_404
        backend._bucket.blob.return_value = mock_blob

        res = await backend.get_embedding("emb:test")
        assert res is None
        metrics.record_gcs_read_failure.assert_not_called()

        # Generic error: record read failure
        mock_blob.download_as_bytes.side_effect = Exception("boom")
        res = await backend.get_embedding("emb:test")
        assert res is None
        metrics.record_gcs_read_failure.assert_called()


@pytest.mark.asyncio
async def test_llm_client_missing_env_var():
    metrics = MagicMock(spec=MetricsCollector)
    config = LLMConfig("http://api", "KEY_VAR_DOES_NOT_EXIST_xyz", "model")
    client = LLMClient(config, metrics)
    with pytest.raises(KeyError, match="Environment variable KEY_VAR_DOES_NOT_EXIST_xyz not found"):
        await client.generate_kv_state("doc")


@pytest.mark.asyncio
async def test_llm_client_http_error():
    metrics = MagicMock(spec=MetricsCollector)
    config = LLMConfig("http://api", "KEY_VAR", "model")

    with patch.dict("os.environ", {"KEY_VAR": "secret"}):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = MagicMock(is_error=True, status_code=500, text="Internal Error")
            client = LLMClient(config, metrics)
            with pytest.raises(RuntimeError, match="LLM call failed: 500"):
                await client.generate_kv_state("doc")


def test_cpu_cache_oversized_and_eviction():
    metrics = MagicMock(spec=MetricsCollector)
    cache = CPUCache(max_embedding_bytes=100, max_kv_bytes=100, metrics=metrics)

    large_vec = np.zeros(1000, dtype=np.float32)
    cache.put_embedding("k1", large_vec)
    assert cache.get_embedding("k1") is None

    small = np.zeros(10, dtype=np.float32)  # 40 bytes
    cache.put_embedding("k1", small)
    cache.put_embedding("k2", small)
    cache.put_embedding("k3", small)  # 120 bytes -> k1 evicted

    assert cache.get_embedding("k1") is None
    assert cache.get_embedding("k2") is not None
    metrics.record_eviction.assert_called_with("embedding")


@pytest.mark.asyncio
async def test_faiss_queue_drop_on_full():
    with patch("embcache._faiss_index.faiss") as mock_faiss:
        # Make _build_index return a MagicMock indexable
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_faiss.IndexFlatIP.return_value = MagicMock()
        mock_faiss.IndexIDMap.return_value = mock_index

        metrics = MagicMock(spec=MetricsCollector)
        config = FAISSIndexConfig(index_type="flat", metric="cosine")

        index = FAISSIndex(config, 128, metrics, max_faiss_write_queue=1)
        # Suppress the writer loop so the queue stays saturated
        index._ensure_writer()
        index._writer_task.cancel()
        await asyncio.gather(index._writer_task, return_exceptions=True)
        # Re-make queue (the cancelled writer may have consumed nothing yet)
        index._write_queue = asyncio.Queue(maxsize=1)

        await index.add("k1", np.zeros(128, dtype=np.float32))
        await index.add("k2", np.zeros(128, dtype=np.float32))  # full → dropped
        metrics.record_faiss_write_dropped.assert_called()


@pytest.mark.asyncio
async def test_faiss_search_empty_index():
    with patch("embcache._faiss_index.faiss") as mock_faiss:
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_faiss.IndexFlatIP.return_value = MagicMock()
        mock_faiss.IndexIDMap.return_value = mock_index

        metrics = MagicMock(spec=MetricsCollector)
        config = FAISSIndexConfig(index_type="flat", metric="cosine")
        index = FAISSIndex(config, 128, metrics)
        res = await index.search(np.zeros(128, dtype=np.float32))
        assert res == []


def test_frozen_fingerprint():
    fp = EmbeddingFingerprint("m1", 128, "h1", "s1", "v1", "pt1", "d1")
    with pytest.raises(FrozenInstanceError):
        fp.model_id = "new"


@pytest.mark.asyncio
async def test_inflight_future_shielding():
    loop = asyncio.get_running_loop()
    f = loop.create_future()

    async def waiter():
        return await asyncio.shield(f)

    t1 = asyncio.create_task(waiter())
    t2 = asyncio.create_task(waiter())

    f.set_result("done")
    assert (await t1) == "done"
    assert (await t2) == "done"
