import os
import logging
from dataclasses import dataclass, field
from typing import Literal, Dict, Any

from ._fingerprint import EmbeddingFingerprint, KVFingerprint

@dataclass
class EmbeddingResult:
    key: str
    embedding: list[float]
    hit: bool
    tier: str  # "exact" | "gpu_l1" | "cpu_l2" | "semantic" | "cold" | "fetch"
    latency_ms: float
    consent_scope: str | None = None
    metadata: Dict | None = None

@dataclass
class KVResult:
    key: str
    kv_state: bytes
    hit: bool
    tier: str  # "gpu_l1" | "cpu_l2" | "cold" | "fetch"
    latency_ms: float
    consent_scope: str | None = None
    metadata: dict | None = None

@dataclass
class FAISSIndexConfig:
    index_type: Literal["flat", "hnsw", "ivf"] = "hnsw"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
    ivf_nlist: int = 100
    metric: Literal["l2", "cosine"] = "cosine"

@dataclass
class LLMConfig:
    endpoint: str
    api_key_env_var: str
    model_id: str
    max_tokens: int = 0
    timeout_seconds: float = 30.0

    def api_key(self) -> str:
        if self.api_key_env_var not in os.environ:
            raise KeyError(f"Environment variable {self.api_key_env_var} not found")
        return os.environ[self.api_key_env_var]

@dataclass
class CacheConfig:
    embedding_fingerprint: EmbeddingFingerprint
    kv_fingerprint: KVFingerprint | None = None
    faiss: FAISSIndexConfig = field(default_factory=FAISSIndexConfig)
    llm: LLMConfig | None = None
    gpu_cache_max_fraction: float = 0.30
    max_faiss_write_queue: int = 100
    context_window: int = 0
    gcs_bucket: str = ""
    gcs_prefix: str = "embcache/"
    local_nvme_path: str = ""
    gds_enabled: bool = False
    enable_prefetch: bool = False
    semantic_similarity_threshold: float = 0.90
    max_embedding_bytes: int = 2 * 1024**3
    max_kv_bytes: int = 8 * 1024**3
    exact_index_max_entries: int = 10000

    def __post_init__(self):
        if not (0.0 < self.gpu_cache_max_fraction < 1.0):
            raise ValueError("gpu_cache_max_fraction must be in (0.0, 1.0)")
        if self.context_window < 0:
            raise ValueError("context_window must be >= 0")
        if not (0.0 <= self.semantic_similarity_threshold <= 1.0):
            raise ValueError("semantic_similarity_threshold must be in [0.0, 1.0]")
        if self.kv_fingerprint is not None and self.llm is None:
            raise ValueError("llm config required when kv_fingerprint is set")
        if self.llm is not None and self.kv_fingerprint is None:
            raise ValueError("kv_fingerprint required when llm config is set")

def _has_nvme() -> bool:
    if os.name == "posix":
        return os.path.exists("/dev/nvme0") or os.path.exists("/dev/nvme0n1")
    return False

def detect_hardware() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        name = torch.cuda.get_device_name(0)
        if "A100" in name and _has_nvme():
            return "gpu_a100"
        return "gpu_other"
    except Exception as e:
        logging.warning(f"Hardware detection failed, falling back to cpu: {e}")
        return "cpu"

def select_tier(hardware: str) -> str:
    return "gpu" if hardware.startswith("gpu") else "cpu"
