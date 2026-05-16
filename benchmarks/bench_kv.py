import asyncio
import time
import argparse
import json
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from embcache import KVCache, MetricsCollector
from ._utils import percentile, append_to_md, ensure_benchmark_md_sections, make_kv_config


async def run_kv_scenario(name, size_bytes, n_bench, simulated_latency_ms):
    dummy_state = b"x" * size_bytes

    async def mock_gen(*_args, **_kwargs):
        await asyncio.sleep(simulated_latency_ms / 1000.0)
        return dummy_state

    mock_llm = MagicMock()
    mock_llm.generate_kv_state = AsyncMock(side_effect=mock_gen)
    mock_llm.close = AsyncMock()

    config = make_kv_config()
    metrics = MetricsCollector(namespace=name)
    cache = KVCache(config, metrics)
    cache._llm = mock_llm

    # Patch GCS to a no-op mock so cold-store probes are local
    cache._gcs = MagicMock()
    cache._gcs.get_kv = AsyncMock(return_value=None)
    cache._gcs.put_kv = AsyncMock(return_value=None)

    # 1. Miss path
    miss_latencies = []
    for i in range(n_bench):
        doc = f"doc_miss_{i}"
        start = time.perf_counter()
        await cache.get_or_fetch_kv(doc)
        miss_latencies.append((time.perf_counter() - start) * 1000)

    # 2. Hit path (CPU L2)
    hit_latencies = []
    for i in range(n_bench):
        doc = f"doc_miss_{i}"
        start = time.perf_counter()
        await cache.get_or_fetch_kv(doc)
        hit_latencies.append((time.perf_counter() - start) * 1000)

    return {
        "scenario": name,
        "kv_state_bytes": size_bytes,
        "cache_miss_p50_ms": percentile(miss_latencies, 50),
        "cache_miss_p95_ms": percentile(miss_latencies, 95),
        "cache_hit_cpu_p50_ms": percentile(hit_latencies, 50),
        "cache_hit_cpu_p95_ms": percentile(hit_latencies, 95),
        "recommended_kv_slot_size_bytes": int(size_bytes * 1.25),
    }


async def run_gpu_kv_scenario(name, size_bytes, n_bench):
    import torch
    from embcache._gpu_cache import GPUCache
    
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        return None
        
    metrics = MetricsCollector(namespace=f"{name}_gpu")
    # Slot size is 1.25x headroom
    gpu_cache = GPUCache(
        embedding_dim=768, 
        kv_slot_size=int(size_bytes * 1.25),
        gpu_cache_max_fraction=0.1,
        embedding_fraction=0.0,
        metrics=metrics
    )
    
    data = b"x" * size_bytes
    # Pre-load 50
    for i in range(50):
        gpu_cache.put_kv(f"gpu_warm_{i}", data)
        
    latencies = []
    for i in range(n_bench):
        key = f"gpu_warm_{np.random.randint(0, 50)}"
        start = time.perf_counter()
        gpu_cache.get_kv(key)
        latencies.append((time.perf_counter() - start) * 1000)
        
    return {
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95)
    }


async def run_gpu_kv_hit_scenario_d():
    import torch
    from embcache._gpu_cache import GPUCache

    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        return {"skipped": True, "results": {}}

    kv_sizes = {"kv_small": 50 * 1024, "kv_medium": 300 * 1024, "kv_large": 600 * 1024}
    n_bench = 200
    results = {}

    for name, size in kv_sizes.items():
        metrics = MetricsCollector(namespace=f"{name}_gpu_hit_d")
        gpu_cache = GPUCache(
            embedding_dim=768,
            kv_slot_size=int(size * 1.25),
            gpu_cache_max_fraction=0.1,
            embedding_fraction=0.0,
            metrics=metrics,
        )

        payload = b"x" * size
        keys = [f"{name}_gpu_warm_{i}" for i in range(50)]
        for key in keys:
            gpu_cache.put_kv(key, payload)

        latencies = []
        for _ in range(n_bench):
            key = keys[np.random.randint(0, len(keys))]
            start = time.perf_counter()
            gpu_cache.get_kv(key)
            latencies.append((time.perf_counter() - start) * 1000)

        results[name] = {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        }

    return {"skipped": False, "results": results}

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bench", type=int, default=100)
    parser.add_argument("--simulated-latency-ms", type=int, default=200)
    parser.add_argument("--output-md", type=str, default="BENCHMARK_RESULTS.md")
    args = parser.parse_args()

    scenarios = [
        ("kv_small", 40 * 1024),
        ("kv_medium", 240 * 1024),
        ("kv_large", 480 * 1024),
    ]

    all_results = []
    gpu_results = {}
    for name, size in scenarios:
        print(f"Running scenario: {name} ({size // 1024} KB)")
        res = await run_kv_scenario(name, size, args.n_bench, args.simulated_latency_ms)
        all_results.append(res)
        
        gpu_res = await run_gpu_kv_scenario(name, size, args.n_bench)
        if gpu_res:
            gpu_results[name] = gpu_res

    scenario_d = await run_gpu_kv_hit_scenario_d()
    if not scenario_d["skipped"]:
        for name, vals in scenario_d["results"].items():
            gpu_results[name] = vals

    print(json.dumps(all_results, indent=2))

    md_content = (
        "\n| Scenario | Size | Miss p50 | Hit p50 (CPU) | GPU Hit p50 ms | GPU Hit p95 ms | Rec. Slot Size |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    for r in all_results:
        gpu_p50 = f"{gpu_results[r['scenario']]['p50']:.2f} ms" if r['scenario'] in gpu_results else "N/A"
        gpu_p95 = f"{gpu_results[r['scenario']]['p95']:.2f} ms" if r['scenario'] in gpu_results else "N/A"
        md_content += (
            f"| {r['scenario']} | {r['kv_state_bytes'] // 1024} KB | "
            f"{r['cache_miss_p50_ms']:.2f} ms | {r['cache_hit_cpu_p50_ms']:.2f} ms | "
            f"{gpu_p50} | {gpu_p95} | "
            f"{r['recommended_kv_slot_size_bytes'] // 1024} KB |\n"
        )

    ensure_benchmark_md_sections(args.output_md)
    append_to_md(args.output_md, "## KV Cache Performance", md_content)


if __name__ == "__main__":
    asyncio.run(main())
