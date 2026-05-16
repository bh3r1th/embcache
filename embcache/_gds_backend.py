"""
GDSBackend: GPU Direct Storage NVMe path.
STATUS: Gated. Activate only after BENCHMARK_RESULTS.md documents
>= 30% latency improvement over GCSBackend baseline.
Set enabled=True in CacheConfig to activate.
v1 uses asyncio executor reads (not true GDS kernel bypass).
True cuFile GDS integration is a post-v1 extension point.
"""

import asyncio
import os
import time
import numpy as np
from typing import Literal

from ._metrics import MetricsCollector, get_logger

_log = get_logger(__name__)

class GDSBackend:
    def __init__(
        self,
        nvme_base_path: str,
        metrics: MetricsCollector,
        enabled: bool = False,
    ):
        self._enabled = enabled
        self._disabled = False
        self._base_path = nvme_base_path
        self.metrics = metrics

        if not enabled:
            _log.info("GDSBackend disabled — benchmark gate not cleared")
            return

        try:
            if not os.path.exists(nvme_base_path):
                _log.error(f"GDS NVMe path {nvme_base_path} does not exist. Disabling.")
                self._disabled = True
                return
            
            os.makedirs(os.path.join(nvme_base_path, "embeddings"), exist_ok=True)
            os.makedirs(os.path.join(nvme_base_path, "kv"), exist_ok=True)
        except Exception as e:
            _log.error(f"GDS initialization error: {e}")
            self._disabled = True

    def _get_path(self, key: str) -> str:
        if key.startswith("emb:"):
            return os.path.join(self._base_path, "embeddings", f"{key[4:]}.npy")
        elif key.startswith("kv:"):
            return os.path.join(self._base_path, "kv", f"{key[3:]}.bin")
        return os.path.join(self._base_path, "misc", key)

    async def get_embedding(self, key: str) -> np.ndarray | None:
        if not self.is_enabled(): return None
        path = self._get_path(key)
        loop = asyncio.get_running_loop()
        
        try:
            t0 = time.monotonic()
            if not await self.exists(key, "embeddings"):
                return None
                
            def _read():
                with open(path, "rb") as f:
                    return f.read()
            
            data = await loop.run_in_executor(None, _read)
            self.metrics.observe_gds_transfer(time.monotonic() - t0)
            return np.frombuffer(data, dtype=np.float32).copy()
        except Exception as e:
            _log.error(f"GDS get_embedding error for {key}: {e}")
            return None

    async def put_embedding(self, key: str, vector: np.ndarray) -> None:
        if not self.is_enabled(): return
        path = self._get_path(key)
        loop = asyncio.get_running_loop()
        
        try:
            data = vector.astype(np.float32).tobytes()
            def _write():
                tmp_path = path + ".tmp"
                with open(tmp_path, "wb") as f:
                    f.write(data)
                os.replace(tmp_path, path)
                
            await loop.run_in_executor(None, _write)
        except Exception as e:
            _log.error(f"GDS put_embedding error for {key}: {e}")

    async def get_kv(self, key: str) -> bytes | None:
        if not self.is_enabled(): return None
        path = self._get_path(key)
        loop = asyncio.get_running_loop()
        
        try:
            t0 = time.monotonic()
            if not await self.exists(key, "kv"):
                return None

            def _read():
                with open(path, "rb") as f:
                    return f.read()
            
            data = await loop.run_in_executor(None, _read)
            self.metrics.observe_gds_transfer(time.monotonic() - t0)
            return data
        except Exception as e:
            _log.error(f"GDS get_kv error for {key}: {e}")
            return None

    async def put_kv(self, key: str, state: bytes) -> None:
        if not self.is_enabled(): return
        path = self._get_path(key)
        loop = asyncio.get_running_loop()
        
        try:
            def _write():
                tmp_path = path + ".tmp"
                with open(tmp_path, "wb") as f:
                    f.write(state)
                os.replace(tmp_path, path)
                
            await loop.run_in_executor(None, _write)
        except Exception as e:
            _log.error(f"GDS put_kv error for {key}: {e}")

    async def exists(self, key: str, pool: Literal["embeddings", "kv"]) -> bool:
        if not self.is_enabled():
            return False
        path = self._get_path(key)
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, lambda: os.path.exists(path))
        except Exception:
            return False

    def is_enabled(self) -> bool:
        return self._enabled and not self._disabled
