import collections
import numpy as np
from typing import Any
from ._metrics import MetricsCollector, get_logger

_log = get_logger(__name__)

class CPUCache:
    def __init__(
        self,
        max_embedding_bytes: int,
        max_kv_bytes: int,
        metrics: MetricsCollector,
    ):
        self.max_embedding_bytes = max_embedding_bytes
        self.max_kv_bytes = max_kv_bytes
        self.metrics = metrics
        
        self.embedding_lru = collections.OrderedDict()
        self.kv_lru = collections.OrderedDict()
        
        self.current_embedding_bytes = 0
        self.current_kv_bytes = 0
        
        self._pinned_attempted = False
        
        if max_embedding_bytes == 0:
            _log.warning("Embedding pool is disabled (max_embedding_bytes=0)")
        if max_kv_bytes == 0:
            _log.warning("KV pool is disabled (max_kv_bytes=0)")

    def _allocate_array(self, vector: np.ndarray) -> np.ndarray:
        if not self._pinned_attempted:
            self._pinned_attempted = True
            try:
                import torch
                # Test allocation to see if pinned memory works
                _ = torch.empty(1, pin_memory=True)
                _log.debug("Using pinned memory via torch for embedding storage")
                self._has_torch = True
            except (ImportError, Exception):
                _log.debug("Falling back to plain numpy for embedding storage (torch pinned memory unavailable)")
                self._has_torch = False

        if self._has_torch:
            try:
                import torch
                t = torch.empty(vector.shape, dtype=torch.float32, pin_memory=True)
                # Copy data accurately
                arr = t.numpy()
                np.copyto(arr, vector)
                return arr
            except Exception:
                return np.array(vector, copy=True, dtype=np.float32)
        return np.array(vector, copy=True, dtype=np.float32)

    def get_embedding(self, key: str) -> np.ndarray | None:
        try:
            if key in self.embedding_lru:
                self.embedding_lru.move_to_end(key)
                self.metrics.record_cpu_l2_hit()
                return np.array(self.embedding_lru[key], copy=True)
            return None
        except Exception as e:
            _log.error(f"Error in get_embedding: {e}")
            return None

    def put_embedding(self, key: str, vector: np.ndarray) -> None:
        try:
            if self.max_embedding_bytes == 0:
                return
            
            size = vector.nbytes
            if size > self.max_embedding_bytes:
                _log.warning(f"Vector too large for CPU cache: {size} > {self.max_embedding_bytes}")
                return

            # Invalidate existing key if present
            self.invalidate(key)

            while self.current_embedding_bytes + size > self.max_embedding_bytes and self.embedding_lru:
                evict_key, evict_val = self.embedding_lru.popitem(last=False)
                self.current_embedding_bytes -= evict_val.nbytes
                self.metrics.record_eviction("embedding")
                _log.debug(f"Evicted embedding {evict_key} from CPU cache")

            self.embedding_lru[key] = self._allocate_array(vector)
            self.current_embedding_bytes += size
            self._update_metrics()
        except Exception as e:
            _log.error(f"Error in put_embedding: {e}")

    def get_kv(self, key: str) -> bytes | None:
        try:
            if key in self.kv_lru:
                self.kv_lru.move_to_end(key)
                self.metrics.record_cpu_l2_hit()
                return self.kv_lru[key]
            return None
        except Exception as e:
            _log.error(f"Error in get_kv: {e}")
            return None

    def put_kv(self, key: str, state: bytes) -> None:
        if isinstance(state, str):
            state = state.encode("utf-8")
        try:
            if self.max_kv_bytes == 0:
                return
            
            size = len(state)
            if size > self.max_kv_bytes:
                _log.warning(f"KV state too large for CPU cache: {size} > {self.max_kv_bytes}")
                return

            self.invalidate(key)

            while self.current_kv_bytes + size > self.max_kv_bytes and self.kv_lru:
                evict_key, evict_val = self.kv_lru.popitem(last=False)
                self.current_kv_bytes -= len(evict_val)
                self.metrics.record_eviction("kv")
                _log.debug(f"Evicted KV state {evict_key} from CPU cache")

            self.kv_lru[key] = state
            self.current_kv_bytes += size
            self._update_metrics()
        except Exception as e:
            _log.error(f"Error in put_kv: {e}")

    def invalidate(self, key: str) -> bool:
        removed = False
        if key in self.embedding_lru:
            val = self.embedding_lru.pop(key)
            self.current_embedding_bytes -= val.nbytes
            removed = True
        if key in self.kv_lru:
            val = self.kv_lru.pop(key)
            self.current_kv_bytes -= len(val)
            removed = True
        if removed:
            self._update_metrics()
        return removed

    def _update_metrics(self) -> None:
        self.metrics.set_slab_bytes(self.current_embedding_bytes, self.current_kv_bytes)
        total_max = self.max_embedding_bytes + self.max_kv_bytes
        if total_max > 0:
            utilization = (self.current_embedding_bytes + self.current_kv_bytes) / total_max
            self.metrics.set_slab_utilization(utilization * 100)

    def stats(self) -> dict:
        return {
            "embedding_count": len(self.embedding_lru),
            "embedding_bytes": self.current_embedding_bytes,
            "embedding_max_bytes": self.max_embedding_bytes,
            "kv_count": len(self.kv_lru),
            "kv_bytes": self.current_kv_bytes,
            "kv_max_bytes": self.max_kv_bytes,
        }
