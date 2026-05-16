import asyncio
import time
from typing import Dict, Optional

from ._config import CacheConfig, KVResult, detect_hardware, select_tier
from ._keys import make_kv_cache_key
from ._cpu_cache import CPUCache
from ._gcs_backend import GCSBackend
from ._gds_backend import GDSBackend
from ._llm_client import LLMClient
from ._metrics import MetricsCollector, get_logger

_log = get_logger(__name__)

class KVCache:
    def __init__(self, config: CacheConfig, metrics: MetricsCollector | None = None):
        if config.kv_fingerprint is None:
            raise ValueError("kv_fingerprint must not be None")
        if config.llm is None:
            raise ValueError("llm config must not be None")

        self.config = config
        self.metrics = metrics or MetricsCollector("default")

        hardware = detect_hardware()
        self._tier = select_tier(hardware)

        self._cpu = CPUCache(
            max_embedding_bytes=0,
            max_kv_bytes=config.max_kv_bytes,
            metrics=self.metrics,
        )

        self._gpu = None
        if self._tier == "gpu":
            try:
                from ._gpu_cache import GPUCache
                self._gpu = GPUCache(
                    embedding_dim=1,
                    kv_slot_size=512 * 1024 * 1024,
                    gpu_cache_max_fraction=config.gpu_cache_max_fraction,
                    embedding_fraction=0.0,
                    metrics=self.metrics,
                )
            except Exception as e:
                _log.warning(f"Failed to initialize GPUCache for KV, falling back to CPU tier: {e}")
                self._tier = "cpu"

        self._gcs = GCSBackend(config.gcs_bucket, config.gcs_prefix, self.metrics)
        self._gds = GDSBackend(
            nvme_base_path=config.local_nvme_path,
            metrics=self.metrics,
            enabled=config.gds_enabled,
        )

        self._llm = LLMClient(config.llm, self.metrics)
        self._inflight: Dict[str, asyncio.Future] = {}
        _log.info(f"KVCache initialized on {self._tier} tier")

    async def get_or_fetch_kv(
        self,
        document: str,
        metadata: Dict | None = None,
    ) -> KVResult:
        t0 = time.monotonic()
        key = make_kv_cache_key(self.config.kv_fingerprint, document)

        if key in self._inflight:
            self.metrics.set_inflight(len(self._inflight))
            _log.info("KV Request coalesced", extra={"key": key, "event": "in_flight_hit"})
            return await asyncio.shield(self._inflight[key])

        future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        self.metrics.set_inflight(len(self._inflight))

        try:
            result = await self._lookup_or_fetch(key, document, t0, metadata)
            if not future.done():
                future.set_result(result)
            return result
        except Exception as e:
            if not future.done():
                future.set_exception(e)
            raise
        finally:
            self._inflight.pop(key, None)
            self.metrics.set_inflight(len(self._inflight))

    async def _lookup_or_fetch(
        self, key: str, document: str, t0: float, metadata: Optional[Dict]
    ) -> KVResult:
        # GPU L1
        if self._gpu:
            hit = self._gpu.get_kv(key)
            if hit is not None:
                self.metrics.record_kv_hit()
                return self._make_kv_result(key, hit, "gpu_l1", t0, metadata)

        # CPU L2
        hit = self._cpu.get_kv(key)
        if hit is not None:
            self.metrics.record_kv_hit()
            if self._gpu:
                self._gpu.put_kv(key, hit)
            return self._make_kv_result(key, hit, "cpu_l2", t0, metadata)

        # Cold (GDS, then GCS)
        for backend in (self._gds, self._gcs):
            hit = await backend.get_kv(key)
            if hit is not None:
                self.metrics.record_kv_hit()
                self._kv_backfill(key, hit)
                return self._make_kv_result(key, hit, "cold", t0, metadata)

        # Miss — generate via LLM
        self.metrics.record_kv_miss()
        kv_state = await self._llm.generate_kv_state(document)
        result = self._make_kv_result(key, kv_state, "fetch", t0, metadata)

        self._kv_backfill(key, kv_state)
        asyncio.create_task(self._gcs.put_kv(key, kv_state))
        return result

    def _kv_backfill(self, key: str, state: bytes) -> None:
        self._cpu.put_kv(key, state)
        if self._gpu:
            self._gpu.put_kv(key, state)

    def _make_kv_result(
        self, key: str, state: bytes, tier: str, t0: float, metadata: Optional[Dict]
    ) -> KVResult:
        _log.info("KV Request served", extra={"key": key, "tier": tier})
        return KVResult(
            key=key,
            kv_state=state,
            hit=(tier != "fetch"),
            tier=tier,
            latency_ms=(time.monotonic() - t0) * 1000,
            consent_scope=metadata.get("consent_scope") if metadata else None,
            metadata=metadata,
        )

    async def warm_from_gcs(self, prefix: str = "") -> int:
        from ._warm import WarmupLoader
        loader = WarmupLoader(self._gcs, None, None, self._cpu, self._gpu, self._tier)
        return await loader.warm_kv(prefix)

    async def close(self) -> None:
        await self._llm.close()
        _log.info("KVCache shutdown")
