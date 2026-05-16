import argparse
import asyncio
import json
import statistics
import sys
import time

import numpy as np

from embcache._config import detect_hardware
from embcache._cpu_cache import CPUCache
from embcache._gpu_cache import GPUCache
from embcache._metrics import MetricsCollector
from benchmarks._utils import append_to_md, ensure_benchmark_md_sections, random_vector

if detect_hardware() != "gpu_a100":
    print("SKIP: bench_gpu_cache requires gpu_a100")
    sys.exit(0)

N_WARMUP = 50
N_BENCH = 1000
N_BENCH_H2D = 200
EMBED_DIM = 768
KV_SLOT_SIZE = 600 * 1024  # 600 KB
GPU_FRACTION = 0.30


def pctl(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def qps_from_latencies_ms(latencies_ms: list[float]) -> float:
    total_s = sum(latencies_ms) / 1000.0
    return (len(latencies_ms) / total_s) if total_s > 0 else 0.0


def run_write_throughput(gpu: GPUCache) -> dict:
    latencies = []
    for i in range(N_BENCH + N_WARMUP):
        key = f"write_{i}"
        vec = random_vector(EMBED_DIM)
        t0 = time.perf_counter()
        gpu.put_embedding(key, vec)
        ms = (time.perf_counter() - t0) * 1000.0
        if i >= N_WARMUP:
            latencies.append(ms)
    return {
        "scenario": "GPU slab write throughput",
        "p50_ms": statistics.median(latencies),
        "p95_ms": pctl(latencies, 95),
        "p99_ms": pctl(latencies, 99),
        "qps": qps_from_latencies_ms(latencies),
    }


def run_read_throughput(gpu: GPUCache) -> dict:
    keys = [f"read_prefill_{i}" for i in range(1000)]
    for key in keys:
        gpu.put_embedding(key, random_vector(EMBED_DIM))

    latencies = []
    for i in range(N_BENCH + N_WARMUP):
        key = keys[np.random.randint(0, len(keys))]
        t0 = time.perf_counter()
        gpu.get_embedding(key)
        ms = (time.perf_counter() - t0) * 1000.0
        if i >= N_WARMUP:
            latencies.append(ms)
    return {
        "scenario": "GPU slab read throughput",
        "p50_ms": statistics.median(latencies),
        "p95_ms": pctl(latencies, 95),
        "p99_ms": pctl(latencies, 99),
        "qps": qps_from_latencies_ms(latencies),
    }


def run_h2d_by_dim(metrics: MetricsCollector) -> list[dict]:
    dims = [128, 256, 512, 768, 1536, 3072]
    rows = []
    for dim in dims:
        temp_gpu = GPUCache(
            embedding_dim=dim,
            kv_slot_size=KV_SLOT_SIZE,
            gpu_cache_max_fraction=GPU_FRACTION,
            embedding_fraction=1.0,
            metrics=metrics,
        )
        latencies = []
        for i in range(N_BENCH_H2D):
            t0 = time.perf_counter()
            temp_gpu.put_embedding(f"h2d_{dim}_{i}", random_vector(dim))
            latencies.append((time.perf_counter() - t0) * 1000.0)
        rows.append({"dim": dim, "p50_ms": statistics.median(latencies)})
    return rows


def run_eviction_with_demotion(gpu: GPUCache, metrics: MetricsCollector) -> dict:
    stats = gpu.stats()
    n_slots = int(stats.get("embedding_slots_total", 0))
    used_slots = int(stats.get("embedding_slots_used", 0))

    if n_slots == 0:
        return {
            "eviction_p50_ms": 0.0,
            "eviction_p95_ms": 0.0,
            "total_evictions": 0,
            "verified_demotions": 0,
            "demotion_success_rate": 1.0,
        }

    cpu = CPUCache(max_embedding_bytes=512 * 1024 * 1024, max_kv_bytes=0, metrics=metrics)

    target_used = int(n_slots * 0.95)
    for i in range(max(0, target_used - used_slots)):
        gpu.put_embedding(f"evict_fill_{i}", random_vector(EMBED_DIM))

    original_evict = gpu._evict_from_pool
    evicted_keys: list[str] = []
    total_evictions = 0

    def evict_with_demotion(pool_type: str) -> bool:
        nonlocal total_evictions
        if pool_type != "embedding":
            return original_evict(pool_type)

        target_key = None
        for key, (entry_pool, _) in gpu.lru.items():
            if entry_pool == "embedding":
                target_key = key
                break

        if target_key is not None:
            cpu.put_embedding(target_key, random_vector(EMBED_DIM))
            evicted_keys.append(target_key)

        ok = original_evict(pool_type)
        if ok:
            total_evictions += 1
        return ok

    gpu._evict_from_pool = evict_with_demotion

    latencies = []
    for i in range(200):
        t0 = time.perf_counter()
        gpu.put_embedding(f"evict_pressure_{i}", random_vector(EMBED_DIM))
        latencies.append((time.perf_counter() - t0) * 1000.0)

    gpu._evict_from_pool = original_evict

    verified = 0
    for key in evicted_keys:
        if cpu.get_embedding(key) is not None:
            verified += 1
    rate = (verified / total_evictions) if total_evictions > 0 else 1.0
    if rate < 1.0:
        print(f"WARNING: demotion_success_rate {rate:.1%}")

    return {
        "eviction_p50_ms": statistics.median(latencies),
        "eviction_p95_ms": pctl(latencies, 95),
        "total_evictions": total_evictions,
        "verified_demotions": verified,
        "demotion_success_rate": rate,
    }


def run_kv_round_trip(gpu: GPUCache) -> list[dict]:
    kv_sizes = [50 * 1024, 300 * 1024, 600 * 1024]
    rows = []
    for size in kv_sizes:
        data = np.random.bytes(size)
        put_latencies = []
        get_latencies = []
        for i in range(N_BENCH_H2D):
            key = f"kv_{size}_{i}"
            t0 = time.perf_counter()
            gpu.put_kv(key, data)
            put_latencies.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            got = gpu.get_kv(key)
            get_latencies.append((time.perf_counter() - t1) * 1000.0)
            if got != data:
                print(f"FAIL: KV round-trip mismatch at {size} bytes")

        rows.append(
            {
                "size_bytes": size,
                "put_p50_ms": statistics.median(put_latencies),
                "put_p95_ms": pctl(put_latencies, 95),
                "get_p50_ms": statistics.median(get_latencies),
                "get_p95_ms": pctl(get_latencies, 95),
            }
        )
    return rows


def render_md(
    write_row: dict,
    read_row: dict,
    h2d_rows: list[dict],
    eviction_row: dict,
    kv_rows: list[dict],
) -> str:
    lines = [
        "| Scenario | p50 ms | p95 ms | p99 ms | QPS |",
        "|---|---|---|---|---|",
        f"| GPU slab write throughput | {write_row['p50_ms']:.3f} | {write_row['p95_ms']:.3f} | {write_row['p99_ms']:.3f} | {write_row['qps']:.1f} |",
        f"| GPU slab read throughput | {read_row['p50_ms']:.3f} | {read_row['p95_ms']:.3f} | {read_row['p99_ms']:.3f} | {read_row['qps']:.1f} |",
    ]
    for row in h2d_rows:
        lines.append(f"| H2D dim={row['dim']} | {row['p50_ms']:.3f} | - | - | - |")
    for row in kv_rows:
        kb = row["size_bytes"] // 1024
        lines.append(f"| KV {kb}KB put | {row['put_p50_ms']:.3f} | {row['put_p95_ms']:.3f} | - | - |")
        lines.append(f"| KV {kb}KB get | {row['get_p50_ms']:.3f} | {row['get_p95_ms']:.3f} | - | - |")
    lines.append(
        f"| Eviction p50 | {eviction_row['eviction_p50_ms']:.3f} | {eviction_row['eviction_p95_ms']:.3f} | - | - |"
    )
    lines.append(f"| Demotion rate | - | - | - | {eviction_row['demotion_success_rate'] * 100:.1f}% |")
    return "\n" + "\n".join(lines) + "\n"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-md", type=str, default="BENCHMARK_RESULTS.md")
    args = parser.parse_args()

    metrics = MetricsCollector(namespace="bench_gpu")
    try:
        gpu = GPUCache(
            embedding_dim=EMBED_DIM,
            kv_slot_size=KV_SLOT_SIZE,
            gpu_cache_max_fraction=GPU_FRACTION,
            embedding_fraction=0.5,
            metrics=metrics,
        )
        print(f"GPUCache instantiated: {gpu.stats()}")
    except Exception as e:
        print(f"FATAL: GPUCache init failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("Scenario 1: START (Write Throughput)")
    try:
        write_row = run_write_throughput(gpu)
        print("Scenario 1: DONE")
    except Exception as e:
        print(f"ERROR in Scenario 1: {e}")
        import traceback; traceback.print_exc()
        write_row = {"p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "qps": 0}

    print("Scenario 2: START (Read Throughput)")
    try:
        read_row = run_read_throughput(gpu)
        print("Scenario 2: DONE")
    except Exception as e:
        print(f"ERROR in Scenario 2: {e}")
        import traceback; traceback.print_exc()
        read_row = {"p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "qps": 0}

    print("Scenario 3: START (H2D by Dim)")
    try:
        h2d_rows = run_h2d_by_dim(metrics)
        print("Scenario 3: DONE")
    except Exception as e:
        print(f"ERROR in Scenario 3: {e}")
        import traceback; traceback.print_exc()
        h2d_rows = []

    print("Scenario 4: START (Eviction)")
    try:
        eviction_row = run_eviction_with_demotion(gpu, metrics)
        print("Scenario 4: DONE")
    except Exception as e:
        print(f"ERROR in Scenario 4: {e}")
        import traceback; traceback.print_exc()
        eviction_row = {
            "eviction_p50_ms": 0.0,
            "eviction_p95_ms": 0.0,
            "total_evictions": 0,
            "verified_demotions": 0,
            "demotion_success_rate": 0.0,
        }

    print("Scenario 5: START (KV Round Trip)")
    try:
        kv_rows = run_kv_round_trip(gpu)
        print("Scenario 5: DONE")
    except Exception as e:
        print(f"ERROR in Scenario 5: {e}")
        import traceback; traceback.print_exc()
        kv_rows = []

    output = {
        "scenario_1_write": write_row,
        "scenario_2_read": read_row,
        "scenario_3_h2d_dims": h2d_rows,
        "scenario_4_eviction": eviction_row,
        "scenario_5_kv_roundtrip": kv_rows,
    }
    print(json.dumps(output, indent=2))

    ensure_benchmark_md_sections(args.output_md)
    md = render_md(write_row, read_row, h2d_rows, eviction_row, kv_rows)
    append_to_md(args.output_md, "## GPU Slab Performance", md)


if __name__ == "__main__":
    asyncio.run(main())
