from dataclasses import FrozenInstanceError
import pytest
from embcache._fingerprint import EmbeddingFingerprint, KVFingerprint

def test_embedding_fingerprint():
    fp = EmbeddingFingerprint("v1", 768, "h1", "s1", "v1", "t1", "d1")
    s1 = fp.to_canonical_string()
    s2 = fp.to_canonical_string()
    assert s1 == s2
    assert "model_id:v1" in s1
    assert "embedding_dim:768" in s1
    
    fp2 = EmbeddingFingerprint("v1", 768, "h1", "s1", "v1", "t1", "d1")
    assert fp.to_canonical_string() == fp2.to_canonical_string()
    
    fp3 = EmbeddingFingerprint("v2", 768, "h1", "s1", "v1", "t1", "d1")
    assert fp.to_canonical_string() != fp3.to_canonical_string()
    
    with pytest.raises(FrozenInstanceError):
        fp.model_id = "new"

def test_kv_fingerprint():
    fp = KVFingerprint("v1", "h1", "t1", "d1")
    assert fp.to_canonical_string() == fp.to_canonical_string()
    
    fp2 = KVFingerprint("v1", "h1", "t1", "d2")
    assert fp.to_canonical_string() != fp2.to_canonical_string()
    
    with pytest.raises(FrozenInstanceError):
        fp.model_id = "new"
    
    assert "llm_endpoint_hash:h1" in fp.to_canonical_string()
