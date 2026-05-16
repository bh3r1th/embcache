import asyncio
from typing import Any, Optional

from ._cpu_cache import CPUCache
from ._exact_index import ExactIndex
from ._faiss_index import FAISSIndex
from ._gcs_backend import GCSBackend
from ._metrics import get_logger

_log = get_logger(__name__)

EMBEDDING_WARM_BATCH = 50
KV_WARM_BATCH = 10


class WarmupLoader:
    def __init__(
        self,
        gcs: GCSBackend,
        exact: Optional[ExactIndex],
        faiss: Optional[FAISSIndex],
        cpu: CPUCache,
        gpu: Optional[Any],
        tier: str,
    ):
        self._gcs = gcs
        self._exact = exact
        self._faiss = faiss
        self._cpu = cpu
        self._gpu = gpu
        self._tier = tier

    async def warm_embeddings(self, prefix: str = "") -> int:
        try:
            keys = await self._gcs.list_keys("embeddings")
        except Exception as e:
            _log.error(f"Failed to list embedding keys from GCS: {e}")
            return 0

        target_keys = [k for k in keys if k.startswith(prefix)]
        success_count = 0

        for i in range(0, len(target_keys), EMBEDDING_WARM_BATCH):
            batch = target_keys[i : i + EMBEDDING_WARM_BATCH]
            results = await asyncio.gather(*(self._gcs.get_embedding(k) for k in batch), return_exceptions=True)

            batch_pairs: list[tuple[str, Any]] = []
            for key, vector in zip(batch, results):
                if isinstance(vector, Exception) or vector is None:
                    _log.warning(f"Failed to warm embedding {key}: {vector}")
                    continue

                try:
                    if self._exact is not None:
                        self._exact.put(key, vector)
                    self._cpu.put_embedding(key, vector)
                    if self._gpu is not None:
                        self._gpu.put_embedding(key, vector)
                    if self._faiss is not None:
                        batch_pairs.append((key, vector))
                    success_count += 1
                except Exception as e:
                    _log.error(f"Error backfilling {key} during warm: {e}")

            if self._faiss is not None and batch_pairs:
                await self._faiss.add_bulk(batch_pairs)

        _log.info(f"Warmed {success_count} embeddings from GCS")
        return success_count

    async def warm_kv(self, prefix: str = "") -> int:
        try:
            keys = await self._gcs.list_keys("kv")
        except Exception as e:
            _log.error(f"Failed to list KV keys from GCS: {e}")
            return 0

        target_keys = [k for k in keys if k.startswith(prefix)]
        success_count = 0

        for i in range(0, len(target_keys), KV_WARM_BATCH):
            batch = target_keys[i : i + KV_WARM_BATCH]
            results = await asyncio.gather(*(self._gcs.get_kv(k) for k in batch), return_exceptions=True)

            for key, state in zip(batch, results):
                if isinstance(state, Exception) or state is None:
                    _log.warning(f"Failed to warm KV state {key}: {state}")
                    continue

                try:
                    self._cpu.put_kv(key, state)
                    if self._gpu is not None:
                        self._gpu.put_kv(key, state)
                    success_count += 1
                except Exception as e:
                    _log.error(f"Error backfilling KV {key} during warm: {e}")

        _log.info(f"Warmed {success_count} KV states from GCS")
        return success_count
