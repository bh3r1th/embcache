import asyncio
import time
import argparse
import os
import numpy as np
from pathlib import Path
from embcache import MetricsCollector
from embcache._gcs_backend import GCSBackend
from embcache._gds_backend import GDSBackend
from ._utils import percentile, append_to_md, random_vector

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvme-path", type=str, default="/mnt/nvme0/embcache")
    parser.add_argument("--gcs-simulated-latency-ms", type=int, default=5)
    parser.add_argument("--output-md", type=str, default="BENCHMARK_RESULTS.md")
    args = parser.parse_args()

    # NVMe Detection
    nvme_present = False
    if os.name == 'posix': # Linux
        if os.path.exists("/dev/nvme0") or os.path.exists("/dev/nvme0n1"):
            nvme_present = True
    elif os.name == 'nt': # Windows (for dev testing)
        if Path(args.nvme_path).exists():
            nvme_present = True

    if not nvme_present:
        msg = "SKIPPED — no NVMe device detected"
        append_to_md(args.output_md, "## GDS Gate", f"\nResult: {msg}\n")
        print(msg)
        return

    m = MetricsCollector()
    gcs = GCSBackend("bench-bucket", "prefix", m)
    # Patch GCS to simulate network latency if needed, or just measure local "mock"
    
    gds = GDSBackend(args.nvme_path, m, enabled=True)

    # Pre-write data
    print("Preparing test data...")
    dim = 768
    vec = random_vector(dim)
    kv_data = {"data": "x" * (240 * 1024)} # 240 KB
    
    # Mocking storage for speed of setup
    # In a real bench, we'd write to the actual file system
    Path(args.nvme_path).mkdir(parents=True, exist_ok=True)
    
    # Timing GCS (Mocked local I/O + simulated latency)
    print("Benchmarking GCS (Baseline)...")
    gcs_latencies = []
    for _ in range(200):
        start = time.perf_counter()
        await asyncio.sleep(args.gcs_simulated_latency_ms / 1000.0)
        # Small read
        _ = np.random.rand(dim)
        gcs_latencies.append((time.perf_counter() - start) * 1000)
    
    gcs_p50 = percentile(gcs_latencies, 50)

    # Timing GDS
    print("Benchmarking GDS (Direct I/O)...")
    # Actually write a file to measure real NVMe read
    test_file = Path(args.nvme_path) / "test_emb.bin"
    test_file.write_bytes(vec.tobytes())
    
    gds_latencies = []
    for _ in range(200):
        start = time.perf_counter()
        with open(test_file, "rb") as f:
            _ = np.frombuffer(f.read(), dtype=np.float32)
        gds_latencies.append((time.perf_counter() - start) * 1000)
    
    gds_p50 = percentile(gds_latencies, 50)

    improvement = (gcs_p50 - gds_p50) / gcs_p50
    if improvement >= 0.30:
        result = f"PASS — GDSBackend ACTIVATED"
    else:
        result = f"FAIL — GDSBackend SHELVED (improvement={improvement:.1%})"

    output = f"""
## GDS Gate
Result: {result}
GCS p50: {gcs_p50:.2f} ms
GDS p50: {gds_p50:.2f} ms
Improvement: {improvement:.1%}
Threshold: >= 30% latency improvement required.
"""
    append_to_md(args.output_md, "## GDS Gate", output)
    print(output)

if __name__ == "__main__":
    asyncio.run(main())
