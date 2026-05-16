"""
GCS Cold Store Backend.
v1 uses a single attempt retry policy. On failure: log and return None/no-op.
All blocking GCS calls are wrapped in loop.run_in_executor to avoid blocking the event loop.
"""

import asyncio
import time
from typing import Literal
import numpy as np

try:
    from google.cloud import storage
except ImportError:
    storage = None

try:
    from google.api_core.exceptions import NotFound
except ImportError:
    class NotFound(Exception):  # fallback when google-api-core not installed
        pass

from ._metrics import MetricsCollector, get_logger

_log = get_logger(__name__)

class GCSBackend:
    def __init__(
        self,
        bucket_name: str,
        prefix: str,
        metrics: MetricsCollector,
    ):
        self._bucket_name = bucket_name
        self._prefix = (prefix.rstrip("/") + "/") if prefix else ""
        self.metrics = metrics
        self._disabled = False

        if storage is None or not bucket_name:
            if storage is None:
                _log.error("google-cloud-storage not installed. GCSBackend disabled.")
            self._disabled = True
            return

        try:
            self._client = storage.Client()
            self._bucket = self._client.bucket(bucket_name)
        except Exception as e:
            _log.error(f"Failed to initialize GCS Client/Bucket: {e}")
            self._disabled = True

    def _get_path(self, key: str) -> str:
        if key.startswith("emb:"):
            return f"{self._prefix}embeddings/{key[4:]}.npy"
        elif key.startswith("kv:"):
            return f"{self._prefix}kv/{key[3:]}.bin"
        return f"{self._prefix}misc/{key}"

    async def get_embedding(self, key: str) -> np.ndarray | None:
        if self._disabled:
            return None
        path = self._get_path(key)
        loop = asyncio.get_running_loop()

        try:
            t0 = time.monotonic()
            blob = self._bucket.blob(path)
            data = await loop.run_in_executor(None, blob.download_as_bytes)
            self.metrics.observe_h2d_transfer(time.monotonic() - t0)
            self.metrics.record_cold_store_hit()
            return np.frombuffer(data, dtype=np.float32).copy()
        except NotFound:
            return None
        except Exception as e:
            if getattr(e, "code", None) == 404:
                return None
            _log.error(f"GCS get_embedding error for {key}: {e}")
            self.metrics.record_gcs_read_failure()
            return None

    async def put_embedding(self, key: str, vector: np.ndarray) -> None:
        if self._disabled:
            return
        path = self._get_path(key)
        loop = asyncio.get_running_loop()

        try:
            data = vector.astype(np.float32).tobytes()
            blob = self._bucket.blob(path)
            await loop.run_in_executor(
                None,
                lambda: blob.upload_from_string(data, content_type="application/octet-stream"),
            )
        except Exception as e:
            _log.error(f"GCS put_embedding error for {key}: {e}")
            self.metrics.record_gcs_write_failure()

    async def get_kv(self, key: str) -> bytes | None:
        if self._disabled:
            return None
        path = self._get_path(key)
        loop = asyncio.get_running_loop()

        try:
            t0 = time.monotonic()
            blob = self._bucket.blob(path)
            data = await loop.run_in_executor(None, blob.download_as_bytes)
            self.metrics.observe_h2d_transfer(time.monotonic() - t0)
            self.metrics.record_cold_store_hit()
            return data
        except NotFound:
            return None
        except Exception as e:
            if getattr(e, "code", None) == 404:
                return None
            _log.error(f"GCS get_kv error for {key}: {e}")
            self.metrics.record_gcs_read_failure()
            return None

    async def put_kv(self, key: str, state: bytes) -> None:
        if self._disabled:
            return
        path = self._get_path(key)
        loop = asyncio.get_running_loop()

        try:
            blob = self._bucket.blob(path)
            await loop.run_in_executor(
                None,
                lambda: blob.upload_from_string(state, content_type="application/octet-stream"),
            )
        except Exception as e:
            _log.error(f"GCS put_kv error for {key}: {e}")
            self.metrics.record_gcs_write_failure()

    async def list_keys(self, pool: Literal["embeddings", "kv"]) -> list[str]:
        if self._disabled:
            return []
        prefix = f"{self._prefix}{pool}/"
        loop = asyncio.get_running_loop()

        try:
            blobs = await loop.run_in_executor(
                None, lambda: list(self._client.list_blobs(self._bucket_name, prefix=prefix))
            )
            keys = []
            for b in blobs:
                name = b.name[len(prefix):]
                if pool == "embeddings" and name.endswith(".npy"):
                    keys.append(f"emb:{name[:-4]}")
                elif pool == "kv" and name.endswith(".bin"):
                    keys.append(f"kv:{name[:-4]}")
            return keys
        except Exception as e:
            _log.error(f"GCS list_keys error: {e}")
            return []

    async def exists(self, key: str, pool: Literal["embeddings", "kv"]) -> bool:
        if self._disabled:
            return False
        path = self._get_path(key)
        loop = asyncio.get_running_loop()
        try:
            blob = self._bucket.blob(path)
            return await loop.run_in_executor(None, blob.exists)
        except Exception:
            return False
