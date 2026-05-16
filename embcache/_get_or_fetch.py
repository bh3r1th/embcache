import asyncio
import time
from typing import Awaitable, Callable, Optional, List, Dict
import numpy as np

from ._config import CacheConfig, EmbeddingResult, detect_hardware, select_tier
from ._keys import make_embedding_cache_key
from ._exact_index import ExactIndex
from ._faiss_index import FAISSIndex
from ._cpu_cache import CPUCache
from ._gcs_backend import GCSBackend
from ._gds_backend import GDSBackend
from ._prefetch import PrefetchEngine
from ._metrics import MetricsCollector, get_logger

_log = get_logger(__name__)

class EmbeddingCache:
    def __init__(self, config: CacheConfig, metrics: MetricsCollector | None = None):
        self.config = config
        self.metrics = metrics or MetricsCollector("default")

        hardware = detect_hardware()
        self._tier = select_tier(hardware)

        self._exact = ExactIndex(max_entries=config.exact_index_max_entries, metrics=self.metrics)
        self._faiss = FAISSIndex(
            config.faiss,
            config.embedding_fingerprint.embedding_dim,
            self.metrics,
            max_faiss_write_queue=config.max_faiss_write_queue,
        )

        self._cpu = CPUCache(
            max_embedding_bytes=config.max_embedding_bytes,
            max_kv_bytes=0,
            metrics=self.metrics,
        )

        self._gpu = None
        if self._tier == "gpu":
            try:
                from ._gpu_cache import GPUCache
                self._gpu = GPUCache(
                    embedding_dim=config.embedding_fingerprint.embedding_dim,
                    kv_slot_size=1024 * 1024,
                    gpu_cache_max_fraction=config.gpu_cache_max_fraction,
                    embedding_fraction=1.0,
                    metrics=self.metrics,
                )
            except Exception as e:
                _log.warning(f"Failed to initialize GPUCache, falling back to CPU tier: {e}")
                self._tier = "cpu"

        self._gcs = GCSBackend(config.gcs_bucket, config.gcs_prefix, self.metrics)
        self._gds = GDSBackend(
            nvme_base_path=config.local_nvme_path,
            metrics=self.metrics,
            enabled=config.gds_enabled,
        )

        self._prefetch = PrefetchEngine(
            fetch_fn=None,
            enabled=config.enable_prefetch,
        ) if config.enable_prefetch else None

        self._inflight: Dict[str, asyncio.Future] = {}
        _log.info(f"EmbeddingCache initialized on {self._tier} tier")

    async def get_or_fetch(
        self,
        query: str,
        fetch_fn: Callable[[str], Awaitable[List[float]]],
        conversation_context: List[str] | None = None,
        metadata: Dict | None = None,
        query_vector: List[float] | None = None,
    ) -> EmbeddingResult:
        t0 = time.monotonic()
        key = make_embedding_cache_key(
            self.config.embedding_fingerprint, query,
            conversation_context, self.config.context_window,
        )

        # Coalesce concurrent identical requests
        if key in self._inflight:
            self.metrics.set_inflight(len(self._inflight))
            _log.info("Request coalesced", extra={"key": key, "event": "in_flight_hit"})
            return await asyncio.shield(self._inflight[key])

        future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        self.metrics.set_inflight(len(self._inflight))

        try:
            result = await self._lookup_or_fetch(key, query, fetch_fn, t0, metadata, query_vector)
            if not future.done():
                future.set_result(result)
            return result
        except Exception as e:
            if not future.done():
                future.set_exception(e)
            _log.error(f"Fetch failure for {key}: {e}")
            raise
        finally:
            self._inflight.pop(key, None)
            self.metrics.set_inflight(len(self._inflight))

    async def _lookup_or_fetch(
        self,
        key: str,
        query: str,
        fetch_fn: Callable[[str], Awaitable[List[float]]],
        t0: float,
        metadata: Optional[Dict],
        query_vector: Optional[List[float]],
    ) -> EmbeddingResult:
        # Exact
        vec = self._exact.get(key)
        if vec is not None:
            return self._make_result(key, vec.tolist(), "exact", t0, metadata)

        # GPU L1
        if self._gpu:
            tvec = self._gpu.get_embedding(key)
            if tvec is not None:
                arr = np.asarray(tvec, dtype=np.float32)
                return self._make_result(key, arr.tolist(), "gpu_l1", t0, metadata)

        # CPU L2
        vec = self._cpu.get_embedding(key)
        if vec is not None:
            if self._gpu:
                self._gpu.put_embedding(key, vec)
            return self._make_result(key, vec.tolist(), "cpu_l2", t0, metadata)

        # Semantic (FAISS) — only if caller supplied a query vector
        if query_vector is not None:
            q_arr = np.asarray(query_vector, dtype=np.float32)
            matches = await self._faiss.search(
                q_arr, top_k=1, threshold=self.config.semantic_similarity_threshold
            )
            if matches:
                matched_key, _score = matches[0]
                near = self._exact.get(matched_key)
                if near is None:
                    near = self._cpu.get_embedding(matched_key)
                if near is None:
                    near = self._faiss.get_vector(matched_key)
                if near is not None:
                    return self._make_result(
                        key, np.asarray(near, dtype=np.float32).tolist(),
                        "semantic", t0, metadata,
                    )

        # Cold storage (NVMe GDS, then GCS)
        for backend in (self._gds, self._gcs):
            vec = await backend.get_embedding(key)
            if vec is not None:
                self._backfill(key, vec)
                return self._make_result(key, vec.tolist(), "cold", t0, metadata)

        # Miss — call user fetch_fn
        vector_list = await fetch_fn(query)
        vec_np = np.asarray(vector_list, dtype=np.float32)
        result = self._make_result(key, list(vector_list), "fetch", t0, metadata)

        self._backfill(key, vec_np)
        asyncio.create_task(self._gcs.put_embedding(key, vec_np))
        return result

    def _backfill(self, key: str, vector: np.ndarray) -> None:
        self._exact.put(key, vector)
        self._cpu.put_embedding(key, vector)
        if self._gpu:
            self._gpu.put_embedding(key, vector)
        asyncio.create_task(self._faiss.add(key, vector))

    def _make_result(
        self, key: str, vector: List[float], tier: str, t0: float, metadata: Optional[Dict]
    ) -> EmbeddingResult:
        _log.info("Request served", extra={"key": key, "tier": tier})
        return EmbeddingResult(
            key=key,
            embedding=vector,
            hit=(tier != "fetch"),
            tier=tier,
            latency_ms=(time.monotonic() - t0) * 1000,
            consent_scope=metadata.get("consent_scope") if metadata else None,
            metadata=metadata,
        )

    async def prime_faiss(self, key: str, vector: List[float]) -> None:
        await self._faiss.add(key, np.array(vector, dtype=np.float32))

    async def warm_from_gcs(self, prefix: str = "") -> int:
        from ._warm import WarmupLoader
        loader = WarmupLoader(self._gcs, self._exact, self._faiss, self._cpu, self._gpu, self._tier)
        return await loader.warm_embeddings(prefix)

    async def close(self) -> None:
        await self._faiss.close()
        _log.info("EmbeddingCache shutdown")
