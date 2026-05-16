from hashlib import sha256
from ._normalize import normalize
from ._fingerprint import EmbeddingFingerprint, KVFingerprint

def make_embedding_cache_key(
    fingerprint: EmbeddingFingerprint,
    query: str,
    context: list[str] | None = None,
    context_window: int = 0,
) -> str:
    context_slice = context[-context_window:] if context and context_window > 0 else []
    payload = fingerprint.to_canonical_string() + "|" + normalize(query)
    if context_slice:
        payload += "|" + "|".join(normalize(c) for c in context_slice)
    return "emb:" + sha256(payload.encode()).hexdigest()

def make_kv_cache_key(
    fingerprint: KVFingerprint,
    document_content: str,
) -> str:
    doc_hash = sha256(document_content.encode()).hexdigest()
    fp_hash = sha256(fingerprint.to_canonical_string().encode()).hexdigest()
    return "kv:" + sha256((doc_hash + "|" + fp_hash).encode()).hexdigest()
