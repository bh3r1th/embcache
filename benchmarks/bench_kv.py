import asyncio
import time
import argparse
import json
from unittest.mock import AsyncMock, MagicMock

from embcache import KVCache, MetricsCollector
from ._utils import percentile, append_to_md, make_kv_config


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
    for name, size in scenarios:
        print(f"Running scenario: {name} ({size // 1024} KB)")
        res = await run_kv_scenario(name, size, args.n_bench, args.simulated_latency_ms)
        all_results.append(res)

    print(json.dumps(all_results, indent=2))

    md_content = (
        "\n| Scenario | Size | Miss p50 | Hit p50 (CPU) | Rec. Slot Size |\n"
        "|---|---|---|---|---|\n"
    )
    for r in all_results:
        md_content += (
            f"| {r['scenario']} | {r['kv_state_bytes'] // 1024} KB | "
            f"{r['cache_miss_p50_ms']:.2f} ms | {r['cache_hit_cpu_p50_ms']:.2f} ms | "
            f"{r['recommended_kv_slot_size_bytes'] // 1024} KB |\n"
        )

    append_to_md(args.output_md, "## KV Cache Performance", md_content)


if __name__ == "__main__":
    asyncio.run(main())
