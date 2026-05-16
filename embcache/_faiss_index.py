import asyncio
import numpy as np
from typing import Tuple

try:
    import faiss
except ImportError:
    faiss = None

from ._config import FAISSIndexConfig
from ._metrics import MetricsCollector, get_logger

_log = get_logger(__name__)

class FAISSIndex:
    def __init__(
        self,
        config: FAISSIndexConfig,
        embedding_dim: int,
        metrics: MetricsCollector,
        max_faiss_write_queue: int = 100,
    ):
        if faiss is None:
            raise ImportError("FAISSIndex requires faiss (faiss-cpu or faiss-gpu)")

        self.config = config
        self.embedding_dim = embedding_dim
        self.metrics = metrics
        self._max_queue = max_faiss_write_queue

        self._index = self._build_index()
        self._trained = (config.index_type != "ivf")

        self._key_to_id = {}
        self._id_to_key = {}
        self._next_id = 0

        # Lazy: queue + writer created when an event loop is running.
        self._write_queue: asyncio.Queue | None = None
        self._write_lock: asyncio.Lock | None = None
        self._writer_task: asyncio.Task | None = None

        _log.info(f"Initialized FAISSIndex ({config.index_type}, metric={config.metric})")

    def _build_index(self):
        dim = self.embedding_dim
        if self.config.index_type == "flat":
            inner = faiss.IndexFlatIP(dim) if self.config.metric == "cosine" else faiss.IndexFlatL2(dim)
            index = faiss.IndexIDMap(inner)
        elif self.config.index_type == "hnsw":
            inner = faiss.IndexHNSWFlat(dim, self.config.hnsw_m)
            inner.hnsw.efConstruction = self.config.hnsw_ef_construction
            inner.hnsw.efSearch = self.config.hnsw_ef_search
            index = faiss.IndexIDMap(inner)
        elif self.config.index_type == "ivf":
            metric = faiss.METRIC_INNER_PRODUCT if self.config.metric == "cosine" else faiss.METRIC_L2
            quantizer = (
                faiss.IndexFlatIP(dim) if self.config.metric == "cosine" else faiss.IndexFlatL2(dim)
            )
            ivf = faiss.IndexIVFFlat(quantizer, dim, self.config.ivf_nlist, metric)
            index = faiss.IndexIDMap(ivf)
        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")

        try:
            import torch
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                _log.info("FAISS using GPU tier")
        except Exception:
            pass

        return index

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _ensure_writer(self) -> None:
        if self._write_queue is None:
            self._write_queue = asyncio.Queue(maxsize=self._max_queue)
            self._write_lock = asyncio.Lock()
        if self._writer_task is None or self._writer_task.done():
            self._writer_task = asyncio.create_task(self._writer_loop())

    async def _writer_loop(self):
        while True:
            try:
                key, vector = await self._write_queue.get()
                async with self._write_lock:
                    ids = np.array([self._next_id], dtype=np.int64)
                    vecs = vector.reshape(1, -1).astype(np.float32)

                    self._index.add_with_ids(vecs, ids)

                    self._key_to_id[key] = self._next_id
                    self._id_to_key[self._next_id] = key
                    self._next_id += 1

                    self.metrics.set_faiss_queue_depth(self._write_queue.qsize())
                self._write_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                _log.error(f"Error in FAISS background writer: {e}")

    async def add(self, key: str, vector: np.ndarray) -> None:
        try:
            self._ensure_writer()
            if not self._trained:
                _log.warning("FAISS IVF index not trained, skipping add")
                return
            if key in self._key_to_id:
                return

            vec = vector
            if self.config.metric == "cosine":
                vec = self._normalize(vec)

            try:
                self._write_queue.put_nowait((key, vec))
                self.metrics.set_faiss_queue_depth(self._write_queue.qsize())
            except asyncio.QueueFull:
                _log.warning("FAISS write queue full, dropping write")
                self.metrics.record_faiss_write_dropped()
        except Exception as e:
            _log.error(f"Error in FAISS add: {e}")

    async def search(
        self, query_vector: np.ndarray, top_k: int = 1, threshold: float = 0.90
    ) -> list[Tuple[str, float]]:
        try:
            self._ensure_writer()
            if self._index.ntotal == 0 or not self._trained:
                return []

            vec = query_vector.reshape(1, -1).astype(np.float32)
            if self.config.metric == "cosine":
                vec = self._normalize(vec.flatten()).reshape(1, -1)

            async with self._write_lock:
                D, I = self._index.search(vec, top_k)

            results = []
            for score, idx in zip(D[0], I[0]):
                if idx == -1:
                    continue
                valid = score >= threshold if self.config.metric == "cosine" else score <= (1.0 - threshold)
                if valid:
                    key = self._id_to_key.get(int(idx))
                    if key:
                        results.append((key, float(score)))
                        self.metrics.record_semantic_hit()
            return results
        except Exception as e:
            _log.error(f"Error in FAISS search: {e}")
            return []

    def get_vector(self, key: str) -> np.ndarray | None:
        try:
            idx = self._key_to_id.get(key)
            if idx is None:
                return None
            recon = self._index.reconstruct(int(idx))
            return np.asarray(recon, dtype=np.float32)
        except Exception:
            return None

    async def train(self, vectors: np.ndarray) -> None:
        if self.config.index_type != "ivf" or self._trained:
            return
        if len(vectors) < self.config.ivf_nlist:
            raise RuntimeError(f"IVF training requires at least {self.config.ivf_nlist} vectors")

        vecs = vectors.astype(np.float32)
        if self.config.metric == "cosine":
            vecs = np.array([self._normalize(v) for v in vecs])

        self._index.train(vecs)
        self._trained = True
        _log.info(f"FAISS IVF index trained on {len(vectors)} vectors")

    async def close(self) -> None:
        if self._writer_task is None:
            return
        self._writer_task.cancel()
        dropped = self._write_queue.qsize() if self._write_queue else 0
        if dropped > 0:
            _log.info(f"Closing FAISSIndex, dropped {dropped} pending writes")
        await asyncio.gather(self._writer_task, return_exceptions=True)

    def __len__(self) -> int:
        return self._index.ntotal

    def stats(self) -> dict:
        s = {
            "index_type": self.config.index_type,
            "metric": self.config.metric,
            "total_vectors": self._index.ntotal,
            "queue_depth": self._write_queue.qsize() if self._write_queue else 0,
        }
        if self.config.index_type == "ivf":
            s["trained"] = self._trained
        return s
