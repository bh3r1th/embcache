import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Any
from embcache import EmbeddingFingerprint, KVFingerprint

def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    return float(np.percentile(data, p))

def append_to_md(path: str, section: str, content: str) -> None:
    p = Path(path)
    if not p.exists():
        p.write_text(f"# Benchmark Results\n\n{section}\n{content}\n", encoding="utf-8")
        return

    lines = p.read_text(encoding="utf-8").splitlines()
    new_lines = []
    found = False
    in_section = False

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == section.strip():
            found = True
            in_section = True
            new_lines.append(line)
            new_lines.append(content.strip())
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            continue

        if line.startswith("## ") and in_section:
            in_section = False

        if not in_section:
            new_lines.append(line)
        i += 1

    if not found:
        print(f"Warning: Section '{section}' not found in {path}. Appending to end.")
        new_lines.append(f"\n{section}")
        new_lines.append(content.strip())

    fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")
    os.replace(tmp_path, path)

def make_test_fingerprint() -> EmbeddingFingerprint:
    return EmbeddingFingerprint(
        model_id="benchmark-model",
        embedding_dim=768,
        tokenizer_hash="bench-hash",
        chunking_strategy_hash="greedy",
        normalization_version="bench-v1",
        prompt_template_hash="bench-pt",
        dataset_version="bench-doc",
    )

def make_test_kv_fingerprint() -> KVFingerprint:
    return KVFingerprint(
        model_id="benchmark-model",
        llm_endpoint_hash="bench-hash",
        prompt_template_hash="bench-pt",
        dataset_version="bench-doc",
    )

def random_vector(dim: int) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v

def make_kv_config(
    llm_endpoint: str = "http://localhost:11434/v1",
    llm_api_key_env_var: str = "EMBCACHE_LLM_API_KEY",
    llm_model_id: str = "gpt-4o",
) -> Any:
    from embcache import CacheConfig, FAISSIndexConfig, LLMConfig
    os.environ.setdefault(llm_api_key_env_var, "test-key-not-used-in-mock")
    return CacheConfig(
        embedding_fingerprint=make_test_fingerprint(),
        kv_fingerprint=KVFingerprint(
            model_id=llm_model_id,
            llm_endpoint_hash="benchmark",
            prompt_template_hash="default",
            dataset_version="v1",
        ),
        faiss=FAISSIndexConfig(),
        llm=LLMConfig(
            endpoint=llm_endpoint,
            api_key_env_var=llm_api_key_env_var,
            model_id=llm_model_id,
            timeout_seconds=30.0,
        ),
        gcs_bucket="",
    )
