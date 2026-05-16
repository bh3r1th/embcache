import collections
from typing import Literal, Any, Dict
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from ._metrics import MetricsCollector, get_logger

_log = get_logger(__name__)

class GPUCache:
    def __init__(
        self,
        embedding_dim: int,
        kv_slot_size: int,
        gpu_cache_max_fraction: float,
        embedding_fraction: float = 0.5,
        metrics: MetricsCollector = None,
    ):
        if torch is None or not torch.cuda.is_available():
            raise ImportError("GPUCache requires CUDA")
        if not (0.0 <= embedding_fraction <= 1.0):
            raise ValueError("embedding_fraction must be in [0.0, 1.0]")

        self.metrics = metrics or MetricsCollector("noop")
        self.embedding_dim = embedding_dim
        self.kv_slot_size = kv_slot_size

        # 1. Slab sizing
        props = torch.cuda.get_device_properties(0)
        self.slab_bytes = int(gpu_cache_max_fraction * props.total_memory)

        # 2. Slot sizing
        self.embedding_slot_size = embedding_dim * 4  # float32

        emb_pool_bytes = int(self.slab_bytes * embedding_fraction)
        kv_pool_bytes = self.slab_bytes - emb_pool_bytes

        self.n_embedding_slots = (emb_pool_bytes // self.embedding_slot_size) if embedding_fraction > 0 else 0
        self.n_kv_slots = (kv_pool_bytes // kv_slot_size) if embedding_fraction < 1.0 else 0

        if self.n_embedding_slots == 0 and self.n_kv_slots == 0:
            raise ValueError(
                f"Slab too small for any pool: emb_slots=0, kv_slots=0, "
                f"slab_mb={self.slab_bytes/1024**2:.1f}"
            )

        # 3. Allocate slab
        self.slab = torch.empty(self.slab_bytes, dtype=torch.uint8, device="cuda:0")
        self._stream = torch.cuda.Stream(device="cuda:0")

        # 4. Pool offsets
        self.embedding_offset = 0
        self.kv_offset = self.n_embedding_slots * self.embedding_slot_size

        # 5. Slot bookkeeping
        self.free_embedding_slots = list(range(self.n_embedding_slots))
        self.free_kv_slots = list(range(self.n_kv_slots))

        # Shared LRU: key -> (pool_type, slot_index)
        self.lru: "collections.OrderedDict[str, tuple[str, int]]" = collections.OrderedDict()

        # Per-slot KV byte length (so reads don't return slot padding)
        self._kv_slot_len: Dict[int, int] = {}

        _log.info(
            f"Initialized GPUCache with {self.n_embedding_slots} embedding slots "
            f"and {self.n_kv_slots} KV slots"
        )

    def _to_float32_cuda(self, vector) -> "torch.Tensor":
        if isinstance(vector, list):
            return torch.tensor(vector, dtype=torch.float32, device="cuda:0")
        if isinstance(vector, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(vector, dtype=np.float32)).to("cuda:0", non_blocking=True)
        # torch.Tensor
        return vector.to(device="cuda:0", dtype=torch.float32, non_blocking=True)

    def get_embedding(self, key: str):
        try:
            if self.n_embedding_slots == 0:
                return None
            if key in self.lru and self.lru[key][0] == "embedding":
                _, slot_idx = self.lru.pop(key)
                self.lru[key] = ("embedding", slot_idx)  # MRU

                offset = self.embedding_offset + (slot_idx * self.embedding_slot_size)

                with torch.cuda.stream(self._stream):
                    data = self.slab[offset : offset + self.embedding_slot_size].view(torch.float32)
                    result = data.cpu()

                self.metrics.record_gpu_l1_hit()
                self._update_gpu_metrics()
                return result
            return None
        except Exception as e:
            _log.error(f"Error in GPU cache get_embedding: {e}")
            return None

    def put_embedding(self, key: str, vector) -> None:
        try:
            if self.n_embedding_slots == 0:
                return

            if torch.cuda.memory_reserved(0) > 0.95 * torch.cuda.get_device_properties(0).total_memory:
                _log.warning("High GPU memory pressure detected")

            src_t = self._to_float32_cuda(vector).contiguous()
            if src_t.numel() * 4 != self.embedding_slot_size:
                _log.warning(
                    f"put_embedding size mismatch: expected {self.embedding_slot_size} bytes, "
                    f"got {src_t.numel() * 4}"
                )
                return

            self.invalidate(key)

            if not self.free_embedding_slots:
                if not self._evict_from_pool("embedding"):
                    _log.warning("Cannot evict from embedding pool; dropping put")
                    return

            slot_idx = self.free_embedding_slots.pop()
            offset = self.embedding_offset + (slot_idx * self.embedding_slot_size)

            with torch.cuda.stream(self._stream):
                self.slab[offset : offset + self.embedding_slot_size].copy_(src_t.view(torch.uint8))

            self.lru[key] = ("embedding", slot_idx)
            self._update_all_metrics()
        except Exception as e:
            _log.error(f"Error in GPU cache put_embedding: {e}")

    def get_kv(self, key: str):
        try:
            if self.n_kv_slots == 0:
                return None
            if key in self.lru and self.lru[key][0] == "kv":
                _, slot_idx = self.lru.pop(key)
                self.lru[key] = ("kv", slot_idx)

                offset = self.kv_offset + (slot_idx * self.kv_slot_size)
                length = self._kv_slot_len.get(slot_idx, self.kv_slot_size)

                with torch.cuda.stream(self._stream):
                    data = self.slab[offset : offset + length]
                    result_cpu = data.cpu()

                self.metrics.record_gpu_l1_hit()
                return bytes(result_cpu.numpy())
            return None
        except Exception as e:
            _log.error(f"Error in GPU cache get_kv: {e}")
            return None

    def put_kv(self, key: str, state: bytes) -> None:
        try:
            if self.n_kv_slots == 0:
                return
            if len(state) > self.kv_slot_size:
                _log.warning(f"KV state too large for GPU slot: {len(state)} > {self.kv_slot_size}")
                return

            self.invalidate(key)

            if not self.free_kv_slots:
                if not self._evict_from_pool("kv"):
                    _log.warning("Cannot evict from kv pool; dropping put")
                    return

            slot_idx = self.free_kv_slots.pop()
            offset = self.kv_offset + (slot_idx * self.kv_slot_size)

            buf = np.frombuffer(state, dtype=np.uint8)
            src_t = torch.from_numpy(buf).to("cuda:0", non_blocking=True)

            with torch.cuda.stream(self._stream):
                self.slab[offset : offset + len(state)].copy_(src_t)

            self.lru[key] = ("kv", slot_idx)
            self._kv_slot_len[slot_idx] = len(state)
            self._update_all_metrics()
        except Exception as e:
            _log.error(f"Error in GPU cache put_kv: {e}")

    def invalidate(self, key: str) -> bool:
        if key in self.lru:
            pool_type, slot_idx = self.lru.pop(key)
            if pool_type == "embedding":
                self.free_embedding_slots.append(slot_idx)
            else:
                self.free_kv_slots.append(slot_idx)
                self._kv_slot_len.pop(slot_idx, None)
            self._update_all_metrics()
            return True
        return False

    def _evict_from_pool(self, pool_type: str) -> bool:
        target_key = None
        for k, (v_pool, _) in self.lru.items():
            if v_pool == pool_type:
                target_key = k
                break

        if target_key is None:
            return False

        _, slot_idx = self.lru.pop(target_key)
        if pool_type == "embedding":
            self.free_embedding_slots.append(slot_idx)
        else:
            self.free_kv_slots.append(slot_idx)
            self._kv_slot_len.pop(slot_idx, None)

        self.metrics.record_eviction(pool_type)
        _log.debug(f"Evicted {pool_type} {target_key} from GPU slot {slot_idx}")
        return True

    def _update_gpu_metrics(self):
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        self.metrics.set_gpu_memory(reserved, allocated)

    def _update_all_metrics(self):
        self._update_gpu_metrics()

        used_emb = self.n_embedding_slots - len(self.free_embedding_slots)
        used_kv = self.n_kv_slots - len(self.free_kv_slots)

        total_slots = self.n_embedding_slots + self.n_kv_slots
        if total_slots > 0:
            self.metrics.set_slab_utilization((used_emb + used_kv) / total_slots * 100)

        emb_bytes = used_emb * self.embedding_slot_size
        kv_bytes = sum(self._kv_slot_len.values())
        self.metrics.set_slab_bytes(emb_bytes, kv_bytes)

    def stats(self) -> dict:
        used_emb = self.n_embedding_slots - len(self.free_embedding_slots)
        used_kv = self.n_kv_slots - len(self.free_kv_slots)
        total_slots = self.n_embedding_slots + self.n_kv_slots
        return {
            "slab_bytes": self.slab_bytes,
            "embedding_slots_total": self.n_embedding_slots,
            "embedding_slots_used": used_emb,
            "kv_slots_total": self.n_kv_slots,
            "kv_slots_used": used_kv,
            "slab_utilization_percent": (used_emb + used_kv) / total_slots * 100 if total_slots > 0 else 0,
        }
