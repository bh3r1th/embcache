# embcache

High-performance two-tier embedding and KV-state cache for healthcare RAG pipelines. Sits between your RAG query pipeline and BigQuery/GCS. GPU-native on A100 (pre-allocated CUDA slab, shared LRU), CPU-native everywhere else. One public interface per cache type.

## Installation

```bash
pip install embcache
# GPU support (faiss-gpu + CUDA slabs)
pip install embcache[gpu]
# Benchmarks and calibration
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
| `max_faiss_write_queue` | `int` | `100` | Background FAISS write-queue depth. Drops writes at ceiling — expected under bulk load. |
| `context_window` | `int` | `0` | Trailing conversation turns folded into embedding key. Benchmark hit rate before increasing. |
| `gcs_bucket` | `str` | `""` | GCS bucket name. Empty string disables GCS cold store. |
| `gcs_prefix` | `str` | `"embcache/"` | Object key prefix in GCS. |
| `local_nvme_path` | `str` | `""` | NVMe base directory for GDS backend. |
| `gds_enabled` | `bool` | `False` | Activate GDS backend. Gated — see GDS Gate below. |
| `enable_prefetch` | `bool` | `False` | Enable co-occurrence prefetch engine. Off by default — see kill criteria. |
| `semantic_similarity_threshold` | `float` | `0.90` | FAISS match cutoff (cosine ≥ threshold, L2 ≤ 1−threshold). |
| `max_embedding_bytes` | `int` | `2 GiB` | CPU L2 budget for embeddings. |
| `max_kv_bytes` | `int` | `8 GiB` | CPU L2 budget for KV states. Tune against real document sizes before production. |
| `exact_index_max_entries` | `int` | `10_000` | LRU capacity for the exact-key index. |

## Result Types

```python
EmbeddingResult(key, embedding, hit, tier, latency_ms, consent_scope, metadata)
KVResult(key, kv_state, hit, tier, latency_ms, consent_scope, metadata)
```

`tier` values: `"exact"`, `"gpu_l1"`, `"cpu_l2"`, `"semantic"`, `"cold"`, `"fetch"`. `hit=True` for every tier except `"fetch"` (caller's `fetch_fn` or LLM generated a fresh value).

## Hardware Tiers

Detected automatically at construction. Never configure manually.

| Detected hardware | What activates |
|---|---|
| A100 + NVMe | GPU L1 slab (CUDA), CPU L2 (pinned), GDS NVMe (if `gds_enabled=True`), GCS cold store |
| Any other CUDA GPU | CPU L2 (pinned), GCS cold store — GPU tier falls back silently |
| CPU only | CPU L2 (numpy LRU), GCS cold store |

FAISS index runs on GPU when CUDA is available, CPU otherwise. Never exposed to caller.

## Observability

Start the Prometheus metrics endpoint before serving traffic:

```python
from embcache import start_metrics_server
start_metrics_server(port=9090)   # scrape at /metrics
```

Key metrics (all labeled by `namespace`):

| Metric | Type | What it measures |
|---|---|---|
| `embcache_exact_hits_total` | Counter | Exact key matches |
| `embcache_semantic_hits_total` | Counter | FAISS ANN matches above threshold |
| `embcache_gpu_l1_hits_total` | Counter | GPU slab hits |
| `embcache_cpu_l2_hits_total` | Counter | CPU LRU hits |
| `embcache_cold_store_hits_total` | Counter | GCS / GDS hits |
| `embcache_kv_hits_total` / `_misses_total` | Counter | KV state cache hits and misses |
| `embcache_evictions_total{entry_type}` | Counter | LRU evictions by type (embedding / kv) |
| `embcache_faiss_writes_dropped_total` | Counter | Writes dropped when queue full |
| `embcache_gcs_write_failures_total` | Counter | GCS write errors |
| `embcache_slab_utilization_percent` | Gauge | GPU slab fill fraction |
| `embcache_slab_embedding_bytes` / `_kv_bytes` | Gauge | Slab usage by entry type |
| `embcache_gpu_memory_reserved_bytes` | Gauge | torch.cuda.memory_reserved |
| `embcache_inflight_coalesced_requests` | Gauge | In-flight deduplicated requests |
| `embcache_faiss_write_queue_depth` | Gauge | Background FAISS queue depth |
| `embcache_kv_generation_latency_seconds` | Histogram | LLM call latency (p50/p95/p99) |
| `embcache_h2d_transfer_latency_seconds` | Histogram | Host-to-device transfer latency |
| `embcache_gds_transfer_latency_seconds` | Histogram | NVMe read latency (GDS path) |

## Benchmarked performance (NVIDIA A100-SXM4-40GB, 83.5 GB RAM)

| Scenario | p50 latency | QPS |
|---|---|---|
| Embedding cache hit (exact) | 0.05 ms | 18,000+ |
| Embedding cache hit (HNSW semantic) | 0.05 ms | 18,456 |
| KV state cache hit (GPU L1, 40 KB) | 0.08 ms | — |
| KV state cache hit (GPU L1, 480 KB) | 0.36 ms | — |
| KV state cache hit (CPU L2, 480 KB) | 0.63 ms | — |
| KV state generation (LLM miss) | ~200 ms | — |
| Overall hit rate (Zipf α=1.2 workload) | — | 98.3% |
| HNSW recall@1 | — | 1.00 |

Full results: [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)

## Feature Flags and Kill Criteria

**Prefetch** (`enable_prefetch=True`): records query co-occurrence, warms predicted neighbors into L1/L2 ahead of access. Disable permanently if pollution rate > 20% or hit rate < 30% after one week of real traffic.

**GDS gate** (`gds_enabled=True` + `local_nvme_path`): bypasses GCS cold store with direct NVMe reads. Off by default. Enable only after running `python -m benchmarks.bench_gds` and confirming ≥ 30% latency improvement over GCS baseline. Document result in `BENCHMARK_RESULTS.md` before enabling.

**Kill criteria** (monitor for first two weeks of production traffic):

| Signal | Threshold | Action |
|---|---|---|
| Overall hit rate | < 20% | Abandon or radically simplify |
| KV generation p95 | > 10s | Gate KVCache behind feature flag, default off |
| Prefetch pollution rate | > 20% | Disable prefetch permanently |
| Prefetch hit rate | < 30% | Disable prefetch permanently |
| GPU vs CPU latency delta | < 15% and GDS gate failed | Revert to CPU-only |

## Limitations (v1)

- **Single event loop only.** Not thread-safe. Async-safe on one event loop per instance.
- **Single CUDA stream per instance.** No multi-stream parallelism.
- **No multi-process cache sharing.** GCS last-write-wins on concurrent writes from multiple processes.
- **No multi-node distribution.** One machine, one cache instance.
- **FAISS index rebuilt on restart.** Not persisted. Use `warm_from_gcs()` at startup to reload from GCS.
- **GPU slab is statically allocated.** Resize requires process restart.
- **KV states are opaque bytes.** Caller owns serialization and deserialization.
- **Semantic lookup requires explicit query vector.** `get_or_fetch` falls through to cold store without one.
- **IVF index requires explicit training.** Call `faiss_index.train(vectors)` with ≥ `ivf_nlist` vectors before serving if using `index_type="ivf"`.

Parked for v2: Redis L0 tier, quantization, patient-level invalidation, FAISS persistence, multi-process sharing, multi-node distribution, true cuFile GDS kernel bypass.
