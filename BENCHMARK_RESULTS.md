# Benchmark Results

## Calibration Recommendations
### System Stats
- CPU RAM: 7.3 GB
- GPU: None
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
    gpu_cache_max_fraction=0.00,
)
```
## Embedding Search Performance
| Scenario | p50 (ms) | p95 (ms) | QPS | Recall@1 |
|---|---|---|---|---|
| cpu_flat | 0.13 | 0.33 | 5550.3 | N/A |
| cpu_hnsw | 0.14 | 0.25 | 5078.7 | 0.00 |
## KV Cache Performance
| Scenario | Size | Miss p50 | Hit p50 (CPU) | Rec. Slot Size |
|---|---|---|---|---|
| kv_small | 40 KB | 202.09 ms | 0.40 ms | 50 KB |
| kv_medium | 240 KB | 205.72 ms | 0.15 ms | 300 KB |
| kv_large | 480 KB | 202.41 ms | 0.14 ms | 600 KB |

## GDS Gate
Result: SKIPPED  no NVMe device detected

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
