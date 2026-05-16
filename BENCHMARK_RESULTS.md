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
Result: SKIPPED — no NVMe device detected
