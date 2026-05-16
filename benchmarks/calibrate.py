import psutil
import shutil
import argparse
import json
import time
import numpy as np
import os
from pathlib import Path
from ._utils import append_to_md

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

def calibrate_hardware():
    results = {}
    
    # 1. Memory
    vm = psutil.virtual_memory()
    results["cpu_ram_total_gb"] = vm.total / (1024**3)
    results["cpu_ram_available_gb"] = vm.available / (1024**3)
    
    # 2. GPU
    if HAS_CUDA:
        props = torch.cuda.get_device_properties(0)
        results["gpu_name"] = props.name
        results["gpu_total_mb"] = props.total_memory / (1024**2)
        results["gpu_free_mb"] = (props.total_memory - torch.cuda.memory_reserved(0)) / (1024**2)
    else:
        results["gpu_name"] = None

    # 3. Disk
    nvme_candidates = [Path("/mnt/nvme0/embcache"), Path("/mnt/nvme/embcache")]
    if os.name == "nt":
        nvme_candidates.append(Path(os.environ.get("EMBCACHE_NVME_PATH", "")))
    nvme_path = next((p for p in nvme_candidates if str(p) and p.exists()), None)
    if nvme_path is not None:
        usage = shutil.disk_usage(nvme_path)
        results["nvme_total_gb"] = usage.total / (1024**3)
        results["nvme_free_gb"] = usage.free / (1024**3)
    else:
        results["nvme_total_gb"] = None

    # 4. Micro-bench
    print("Running micro-benchmarks...")
    start = time.perf_counter()
    for _ in range(10):
        _ = np.zeros(512 * 1024 * 1024 // 4, dtype=np.float32) # 512 MiB
    cpu_alloc_latency = (time.perf_counter() - start) / 10 * 1000
    results["cpu_alloc_512mib_ms"] = cpu_alloc_latency

    if HAS_CUDA:
        start = time.perf_counter()
        for _ in range(10):
            t = torch.zeros(512 * 1024 * 1024 // 4, device="cuda")
            torch.cuda.synchronize()
        gpu_alloc_latency = (time.perf_counter() - start) / 10 * 1000
        results["gpu_alloc_512mib_ms"] = gpu_alloc_latency

    # Recommendations
    rec = {}
    if HAS_CUDA:
        rec["gpu_cache_max_fraction"] = min(0.30, (results["gpu_free_mb"] * 0.8) / results["gpu_total_mb"])
    
    rec["max_embedding_bytes"] = int(vm.total * 0.10)
    rec["max_kv_bytes"] = int(vm.total * 0.20)
    
    return results, rec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-md", type=str, default="BENCHMARK_RESULTS.md")
    args = parser.parse_args()

    results, rec = calibrate_hardware()
    
    snippet = f"""
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
    gpu_cache_max_fraction={rec.get('gpu_cache_max_fraction', 0.0):.2f},
)
"""
    print(snippet)
    
    md_content = f"""
### System Stats
- CPU RAM: {results['cpu_ram_total_gb']:.1f} GB
- GPU: {results['gpu_name'] or 'None'}
- NVMe: {'Detected' if results['nvme_total_gb'] else 'Not Found'}

### Recommended Config
```python
{snippet.strip()}
```
"""
    append_to_md(args.output_md, "## Calibration Recommendations", md_content)

if __name__ == "__main__":
    main()
