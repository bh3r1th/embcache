import argparse
import asyncio
import hashlib
import json
import random
import statistics
from collections import Counter

import numpy as np

from embcache import CacheConfig, EmbeddingCache, EmbeddingResult, MetricsCollector
from embcache._keys import make_embedding_cache_key
from benchmarks._utils import (
    append_to_md,
    ensure_benchmark_md_sections,
    make_test_fingerprint,
    random_vector,
)

N_UNIQUE_QUERIES = 1000
N_REQUESTS = 5000
ZIPF_ALPHA = 1.2
EMBED_DIM = 768
RANDOM_SEED = 42


def make_fetch_fn(query_to_vector):
    async def fetch_fn(query):
        base_query = query[:-4] if query.endswith(" the") else query
        return query_to_vector[base_query].tolist()

    return fetch_fn


def p50(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=N_UNIQUE_QUERIES)
    parser.add_argument("--n-requests", type=int, default=N_REQUESTS)
    parser.add_argument("--output-md", type=str, default="BENCHMARK_RESULTS.md")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    queries = [f"query_{i:04d}" for i in range(args.n_queries)]
    weights = [1.0 / (i ** ZIPF_ALPHA) for i in range(1, args.n_queries + 1)]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    sampled = random.choices(queries, weights=weights, k=args.n_requests)

    query_to_vector = {q: random_vector(EMBED_DIM) for q in queries}
    fetch_fn = make_fetch_fn(query_to_vector)

    config = CacheConfig(
        embedding_fingerprint=make_test_fingerprint(),
        gcs_bucket="",
        context_window=0,
    )
    cache = EmbeddingCache(config, MetricsCollector(namespace="bench_hitrate"))

    top_100 = queries[:100]
    for q in top_100:
        await cache.get_or_fetch(q, fetch_fn)

    # After pre-warm loop
    bulk_items = [
        (make_embedding_cache_key(config.embedding_fingerprint, q), query_to_vector[q])
        for q in top_100
    ]
    await cache._faiss.add_bulk(bulk_items)
    print(f"FAISS bulk loaded {len(bulk_items)} vectors")

    exact_hits = 0
    cpu_l2_hits = 0
    semantic_hits = 0
    cold_misses = 0
    semantic_attempts = 0
    semantic_attempt_cold = 0
    tier_counter = Counter()
    per_tier_latencies: dict[str, list[float]] = {
        "exact": [],
        "cpu_l2": [],
        "semantic": [],
        "cold": [],
    }

    for i, q in enumerate(sampled):
        query = q
        if (i + 1) % 5 == 0:
            query = f"{q} the"
            semantic_attempts += 1

        qvec = query_to_vector[q]
        result: EmbeddingResult = await cache.get_or_fetch(query, fetch_fn, query_vector=qvec.tolist())
        tier = result.tier
        tier_counter[tier] += 1

        if tier == "exact":
            exact_hits += 1
            per_tier_latencies["exact"].append(result.latency_ms)
        elif tier == "cpu_l2":
            cpu_l2_hits += 1
            per_tier_latencies["cpu_l2"].append(result.latency_ms)
        elif tier == "semantic":
            semantic_hits += 1
            per_tier_latencies["semantic"].append(result.latency_ms)
        elif tier == "cold":
            cold_misses += 1
            per_tier_latencies["cold"].append(result.latency_ms)
        elif tier == "fetch":
            cold_misses += 1

        if query.endswith(" the"):
            if tier == "cold" or tier == "fetch":
                semantic_attempt_cold += 1

    total = len(sampled)
    overall_hit_rate = (total - cold_misses) / total if total > 0 else 0.0
    exact_hit_rate = exact_hits / total if total > 0 else 0.0
    cpu_l2_hit_rate = cpu_l2_hits / total if total > 0 else 0.0
    semantic_hit_rate = semantic_hits / semantic_attempts if semantic_attempts > 0 else 0.0
    cold_miss_rate = cold_misses / total if total > 0 else 0.0

    if overall_hit_rate < 0.20:
        print(f"WARNING: hit rate {overall_hit_rate:.1%} below kill threshold 20%")
    else:
        print(f"OK: hit rate {overall_hit_rate:.1%}")

    output = {
        "requests_total": total,
        "n_unique_queries": args.n_queries,
        "zipf_alpha": ZIPF_ALPHA,
        "seed_hash": hashlib.sha256(str(RANDOM_SEED).encode("utf-8")).hexdigest()[:8],
        "tier_counts": dict(tier_counter),
        "overall_hit_rate": overall_hit_rate,
        "exact_hit_rate": exact_hit_rate,
        "cpu_l2_hit_rate": cpu_l2_hit_rate,
        "semantic_hit_rate": semantic_hit_rate,
        "cold_miss_rate": cold_miss_rate,
        "semantic_attempts": semantic_attempts,
        "semantic_attempt_cold": semantic_attempt_cold,
        "p50_latency_ms": {
            "exact": p50(per_tier_latencies["exact"]),
            "cpu_l2": p50(per_tier_latencies["cpu_l2"]),
            "cold": p50(per_tier_latencies["cold"]),
        },
        "kill_criterion_pass": overall_hit_rate >= 0.20,
    }
    print(json.dumps(output, indent=2))

    md_lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Overall hit rate | {overall_hit_rate * 100:.1f}% |",
        f"| Exact hit rate | {exact_hit_rate * 100:.1f}% |",
        f"| CPU L2 hit rate | {cpu_l2_hit_rate * 100:.1f}% |",
        f"| Semantic hit rate | {semantic_hit_rate * 100:.1f}% |",
        f"| Cold miss rate | {cold_miss_rate * 100:.1f}% |",
        f"| p50 latency (exact) | {p50(per_tier_latencies['exact']):.3f} ms |",
        f"| p50 latency (cpu_l2) | {p50(per_tier_latencies['cpu_l2']):.3f} ms |",
        f"| p50 latency (cold) | {p50(per_tier_latencies['cold']):.3f} ms |",
        f"| Kill criterion (>20%) | {'PASS' if overall_hit_rate >= 0.20 else 'FAIL'} |",
    ]
    ensure_benchmark_md_sections(args.output_md)
    append_to_md(args.output_md, "## Hit Rate Validation", "\n" + "\n".join(md_lines) + "\n")
    await cache.close()


if __name__ == "__main__":
    asyncio.run(main())
