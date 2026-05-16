# Benchmark Results

## Calibration Recommendations
### System Stats
- CPU RAM: 83.5 GB
- GPU: NVIDIA A100-SXM4-40GB
- NVMe: Not Found

### Recommended Config
```python
# Recommended CacheConfig for detected hardware:
config = CacheConfig(
    embedding_fingerprint=EmbeddingFingerprint(
        model_id="your-model",
        embedding_dim=768,
        tokenizer_hash="...",
        chunking_strategy_hash="...",
        normalization_version="...",
        prompt_template_hash="...",
        dataset_version="..."
    ),
    gpu_cache_max_fraction=0.30,
)
```

## Embedding Search Performance
| Scenario | p50 (ms) | p95 (ms) | QPS | Recall@1 |
|---|---|---|---|---|
| cpu_flat | 0.05 | 0.06 | 18096.5 | N/A |
| cpu_hnsw | 0.05 | 0.06 | 18138.2 | 1.00 |
| gpu_flat | 0.05 | 0.06 | 18187.5 | N/A |
| gpu_hnsw | 0.05 | 0.05 | 18456.1 | 1.00 |

## KV Cache Performance
| Scenario | Size | Miss p50 | Hit p50 (CPU) | GPU Hit p50 ms | GPU Hit p95 ms | Rec. Slot Size |
|---|---|---|---|---|---|---|
| kv_small | 40 KB | 201.58 ms | 0.45 ms | 0.08 ms | 0.11 ms | 50 KB |
| kv_medium | 240 KB | 201.66 ms | 0.53 ms | 0.22 ms | 0.24 ms | 300 KB |
| kv_large | 480 KB | 201.84 ms | 0.63 ms | 0.36 ms | 0.39 ms | 600 KB |

## GDS Gate
Result: SKIPPED — no NVMe device detected
Threshold: >= 30% latency improvement over GCS baseline required to enable GDSBackend.

## GPU Slab Performance
| Scenario | p50 ms | p95 ms | p99 ms | QPS |
|---|---|---|---|---|
| GPU slab write throughput | 0.530 | 0.935 | 0.999 | 1705.1 |
| GPU slab read throughput | 0.323 | 0.708 | 0.749 | 2774.4 |
| H2D dim=128 | 0.585 | - | - | - |
| H2D dim=256 | 0.585 | - | - | - |
| H2D dim=512 | 0.593 | - | - | - |
| H2D dim=768 | 0.604 | - | - | - |
| H2D dim=1536 | 0.632 | - | - | - |
| H2D dim=3072 | 0.678 | - | - | - |
| KV 50KB put | 0.386 | 0.815 | - | - |
| KV 50KB get | 0.121 | 0.148 | - | - |
| KV 300KB put | 0.516 | 1.007 | - | - |
| KV 300KB get | 0.231 | 0.268 | - | - |
| KV 600KB put | 0.637 | 1.226 | - | - |
| KV 600KB get | 0.367 | 0.406 | - | - |
| Eviction p50 | 0.594 | 1.033 | - | - |
| Demotion rate | - | - | - | 100.0% |

## Hit Rate Validation
| Metric | Value |
|---|---|
| Overall hit rate | 98.3% |
| Exact hit rate | 82.0% |
| CPU L2 hit rate | 0.0% |
| Semantic hit rate | 81.3% |
| Cold miss rate | 1.7% |
| p50 latency (exact) | 0.052 ms |
| p50 latency (cpu_l2) | 0.000 ms |
| p50 latency (cold) | 0.000 ms |
| Kill criterion (>20%) | PASS |
