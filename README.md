# embcache

High-performance multi-tier embedding and KV-state cache for LLM orchestration. Provides low-latency retrieval across CPU (L2), GPU (L1), and Cloud Storage tiers, with FAISS-based semantic similarity search and optional NVMe/GDS fast-recovery for KV state.

## Installation

```bash
pip install embcache
# GPU support (faiss-gpu + CUDA slabs)
pip install embcache[gpu]
# Benchmarks
pip install embcache[bench]
```

## Usage

### 1. Embedding Cache

```python
import asyncio
from embcache import (
    EmbeddingCache, CacheConfig, EmbeddingFingerprint,
    FAISSIndexConfig, MetricsCollector,
)

fingerprint = EmbeddingFingerprint(
    model_id="text-embedding-3-small",
    embedding_dim=1536,
    tokenizer_hash="...",
    chunking_strategy_hash="...",
    normalization_version="v1",
    prompt_template_hash="...",
    dataset_version="2026-05",
)

config = CacheConfig(
    embedding_fingerprint=fingerprint,
    faiss=FAISSIndexConfig(index_type="hnsw"),
    gcs_bucket="my-cache-bucket",
)

async def my_embed(text: str) -> list[float]:
    # call your real embedding model here
    ...

async def main():
    cache = EmbeddingCache(config, MetricsCollector(namespace="prod"))
    result = await cache.get_or_fetch("query text", fetch_fn=my_embed)
    # result.embedding, result.hit, result.tier, result.latency_ms
    await cache.close()

asyncio.run(main())
```

### 2. KV-State Cache

```python
import asyncio
from embcache import (
    KVCache, CacheConfig, EmbeddingFingerprint, KVFingerprint, LLMConfig,
)

config = CacheConfig(
    embedding_fingerprint=EmbeddingFingerprint(...),  # required
    kv_fingerprint=KVFingerprint(
        model_id="gpt-4o",
        llm_endpoint_hash="sha256(endpoint+model)",
        prompt_template_hash="...",
        dataset_version="2026-05",
    ),
    llm=LLMConfig(
        endpoint="https://api.openai.com/v1",
        api_key_env_var="OPENAI_API_KEY",
        model_id="gpt-4o",
    ),
    gcs_bucket="my-cache-bucket",
)

async def main():
    cache = KVCache(config)
    result = await cache.get_or_fetch_kv("prompt or document text")
    # result.kv_state, result.hit, result.tier, result.latency_ms
    await cache.close()

asyncio.run(main())
```

## `CacheConfig` Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `embedding_fingerprint` | `EmbeddingFingerprint` | required | Versioning for embedding model + strategy. |
| `kv_fingerprint` | `KVFingerprint \| None` | `None` | Versioning for LLM + prompt templates. Required iff `llm` is set. |
| `llm` | `LLMConfig \| None` | `None` | Endpoint config for KV-state generation. |
| `faiss` | `FAISSIndexConfig` | `hnsw` defaults | Semantic-tier index parameters. |
| `gpu_cache_max_fraction` | `float` | `0.30` | Fraction of VRAM reserved for L1 slab (open interval `(0, 1)`). |
| `max_faiss_write_queue` | `int` | `100` | Background FAISS write-queue depth. |
| `context_window` | `int` | `0` | Number of trailing conversation-context strings folded into the embedding key. |
| `gcs_bucket` | `str` | `""` | GCS bucket name (empty disables GCS). |
| `gcs_prefix` | `str` | `"embcache/"` | Object key prefix in GCS. |
| `local_nvme_path` | `str` | `""` | NVMe base directory for GDS backend. |
| `gds_enabled` | `bool` | `False` | Activate GDS backend (gated on benchmark improvement). |
| `enable_prefetch` | `bool` | `False` | Enable co-occurrence prefetch engine. |
| `semantic_similarity_threshold` | `float` | `0.90` | FAISS match cutoff (cosine ≥ thr, L2 ≤ 1−thr). |
| `max_embedding_bytes` | `int` | `2 GiB` | CPU L2 budget for embeddings. |
| `max_kv_bytes` | `int` | `8 GiB` | CPU L2 budget for KV state. |
| `exact_index_max_entries` | `int` | `10000` | LRU size for the exact-key index. |

## Result Types

```python
EmbeddingResult(key, embedding, hit, tier, latency_ms, consent_scope, metadata)
KVResult(key, kv_state, hit, tier, latency_ms, consent_scope, metadata)
```

Tiers: `"exact"`, `"gpu_l1"`, `"cpu_l2"`, `"semantic"`, `"cold"`, `"fetch"`. `hit` is `True` for every tier except `"fetch"` (which means the caller's `fetch_fn` / LLM generated a fresh value).

## Hardware Tiers

Hardware detected at construction:
- **CPU** (default): activates `CPUCache` (LRU) and `FAISSIndex` (CPU).
- **GPU** (any CUDA device): activates `GPUCache` (preallocated slab) and routes FAISS to `StandardGpuResources` when available.

GDS NVMe path stays gated behind `gds_enabled=True` and `local_nvme_path` until the benchmark suite proves ≥30% latency improvement over GCS baseline.

## Observability

```python
from embcache import start_metrics_server
start_metrics_server(port=9090)
```

Key Prometheus metrics (all labeled by `namespace`):
- `embcache_exact_hits_total`, `embcache_gpu_l1_hits_total`, `embcache_cpu_l2_hits_total`, `embcache_semantic_hits_total`, `embcache_cold_store_hits_total`, `embcache_kv_hits_total`, `embcache_kv_misses_total`
- `embcache_evictions_total{entry_type}`
- `embcache_gcs_read_failures_total`, `embcache_gcs_write_failures_total`, `embcache_faiss_writes_dropped_total`
- `embcache_slab_utilization_percent`, `embcache_slab_embedding_bytes`, `embcache_slab_kv_bytes`
- `embcache_gpu_memory_reserved_bytes`, `embcache_gpu_memory_allocated_bytes`
- `embcache_inflight_coalesced_requests`, `embcache_faiss_write_queue_depth`
- Histograms: `embcache_kv_generation_latency_seconds`, `embcache_kv_state_size_bytes`, `embcache_gds_transfer_latency_seconds`, `embcache_h2d_transfer_latency_seconds`

## Feature Flags & Gates

- **Prefetch** (`enable_prefetch=True`): records co-occurrence and warms semantic neighbors into L1/L2 ahead of access.
- **GDS Gate** (`gds_enabled=True` + `local_nvme_path`): bypasses GCS to NVMe; off until benchmarks show ≥30% latency improvement.
- **Versioning**: any fingerprint change yields a different cache key — no silent stale hits.

## Limitations

- **Concurrency**: FAISS writes use a single `asyncio.Lock`; not tuned for 10k+ concurrent writes/sec without batching.
- **Persistence**: GCS is the durable tier; local FAISS index must be rebuilt or warmed (`EmbeddingCache.warm_from_gcs`) on restart.
- **Schema**: KV states are opaque bytes — caller owns serialization.
- **Memory**: GPU slab is statically allocated; resize requires process restart.
- **Semantic lookup**: `get_or_fetch` performs FAISS search only when the caller passes an explicit `query_vector`. Without it the request falls through to cold storage / fetch_fn.
