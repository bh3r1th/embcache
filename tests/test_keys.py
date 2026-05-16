import pytest
from embcache._keys import make_embedding_cache_key, make_kv_cache_key
from embcache._fingerprint import EmbeddingFingerprint, KVFingerprint

def test_make_embedding_cache_key():
    fp = EmbeddingFingerprint("m1", 768, "t1", "s1", "v1", "pt1", "d1")
    q = "test query"
    k1 = make_embedding_cache_key(fp, q)
    k2 = make_embedding_cache_key(fp, q)
    assert k1 == k2
    assert k1.startswith("emb:")
    assert len(k1) == 68
    
    k3 = make_embedding_cache_key(fp, "other")
    assert k1 != k3
    
    fp2 = EmbeddingFingerprint("m2", 768, "t1", "s1", "v1", "pt1", "d1")
    k4 = make_embedding_cache_key(fp2, q)
    assert k1 != k4
    
    ctx = ["c1", "c2"]
    ka = make_embedding_cache_key(fp, q, context=ctx, context_window=0)
    kb = make_embedding_cache_key(fp, q, context=ctx, context_window=2)
    assert ka != kb
    
    kc = make_embedding_cache_key(fp, q, context=ctx, context_window=10)
    kd = make_embedding_cache_key(fp, q, context=ctx, context_window=2)
    assert kc == kd

def test_make_kv_cache_key():
    fp = KVFingerprint("m1", "e1", "pt1", "d1")
    doc = "content"
    k1 = make_kv_cache_key(fp, doc)
    k2 = make_kv_cache_key(fp, doc)
    assert k1 == k2
    assert k1.startswith("kv:")
    assert len(k1) == 67
    
    k3 = make_kv_cache_key(fp, "different")
    assert k1 != k3
    
    fp2 = KVFingerprint("m1", "e1", "pt1", "d2")
    k4 = make_kv_cache_key(fp2, doc)
    assert k1 != k4
