import asyncio
import time
import argparse
import json
import numpy as np

from embcache import (
    EmbeddingCache, CacheConfig, FAISSIndexConfig, MetricsCollector,
)
from ._utils import percentile, append_to_md, make_test_fingerprint, random_vector

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


async def run_scenario(name, config, n_vectors, dim, n_bench, n_warmup=50):
    metrics = MetricsCollector(namespace=name)
    cache = EmbeddingCache(config, metrics)

    print(f"[{name}] Prefilling {n_vectors} vectors...")
    # Prefill using the same key derivation get_or_fetch uses, by priming
    # exact/cpu/faiss directly under deterministic query strings.
    from embcache._keys import make_embedding_cache_key
    query_strings = [f"query_{i}" for i in range(n_vectors)]
    prefilled_vecs = [random_vector(dim) for _ in range(n_vectors)]
    for q, vec in zip(query_strings, prefilled_vecs):
        key = make_embedding_cache_key(config.embedding_fingerprint, q)
        cache._exact.put(key, vec)
        cache._cpu.put_embedding(key, vec)
        await cache._faiss.add(key, vec)

    async def dummy_fetch(text: str):
        return random_vector(dim).tolist()

    print(f"[{name}] Warming up...")
    for _ in range(n_warmup):
        idx = int(np.random.randint(n_vectors))
        await cache.get_or_fetch(query_strings[idx], fetch_fn=dummy_fetch)

    print(f"[{name}] Starting benchmark...")
    latencies = []
    start_total = time.perf_counter()
    correct = 0

    for _ in range(n_bench):
        idx = int(np.random.randint(n_vectors))
        q_text = query_strings[idx]

        start = time.perf_counter()
        res = await cache.get_or_fetch(q_text, fetch_fn=dummy_fetch)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        if name.endswith("_hnsw") and res.hit:
            correct += 1

    total_time = time.perf_counter() - start_total

    await cache.close()

    return {
        "scenario": name,
        "n_vectors": n_vectors,
        "embedding_dim": dim,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "throughput_qps": n_bench / total_time if total_time > 0 else 0,
        "recall_at_1": (correct / n_bench) if "_hnsw" in name else None,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--n-vectors", type=int, default=10000)
    parser.add_argument("--n-bench", type=int, default=500)
    parser.add_argument("--output-md", type=str, default="BENCHMARK_RESULTS.md")
    args = parser.parse_args()

    all_results = []
    fp = make_test_fingerprint()

    cfg1 = CacheConfig(
        embedding_fingerprint=fp,
        faiss=FAISSIndexConfig(index_type="flat", metric="cosine"),
        gcs_bucket="",
    )
    all_results.append(await run_scenario("cpu_flat", cfg1, args.n_vectors, args.dim, args.n_bench))

    cfg2 = CacheConfig(
        embedding_fingerprint=fp,
        faiss=FAISSIndexConfig(index_type="hnsw", hnsw_m=32, metric="cosine"),
        gcs_bucket="",
    )
    all_results.append(await run_scenario("cpu_hnsw", cfg2, args.n_vectors, args.dim, args.n_bench))

    if HAS_CUDA:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name.lower()
        if "a100" in gpu_name or "h100" in gpu_name or "rtx" in gpu_name:
            cfg3 = CacheConfig(
                embedding_fingerprint=fp,
                faiss=FAISSIndexConfig(index_type="flat", metric="cosine"),
                gcs_bucket="",
            )
            all_results.append(await run_scenario("gpu_flat", cfg3, args.n_vectors, args.dim, args.n_bench))

            cfg4 = CacheConfig(
                embedding_fingerprint=fp,
                faiss=FAISSIndexConfig(index_type="hnsw", metric="cosine"),
                gcs_bucket="",
            )
            all_results.append(await run_scenario("gpu_hnsw", cfg4, args.n_vectors, args.dim, args.n_bench))
        else:
            print(f"Skipping GPU benchmarks: {props.name} not target hardware.")
    else:
        print("Skipping GPU benchmarks: No CUDA available.")

    print(json.dumps(all_results, indent=2))

    md_content = "\n| Scenario | p50 (ms) | p95 (ms) | QPS | Recall@1 |\n|---|---|---|---|---|\n"
    for r in all_results:
        recall = f"{r['recall_at_1']:.2f}" if r["recall_at_1"] is not None else "N/A"
        md_content += (
            f"| {r['scenario']} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | "
            f"{r['throughput_qps']:.1f} | {recall} |\n"
        )

    append_to_md(args.output_md, "## Embedding Search Performance", md_content)


if __name__ == "__main__":
    asyncio.run(main())
