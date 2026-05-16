from dataclasses import dataclass, fields

@dataclass(frozen=True)
class EmbeddingFingerprint:
    model_id: str
    embedding_dim: int
    tokenizer_hash: str
    chunking_strategy_hash: str
    normalization_version: str
    prompt_template_hash: str
    dataset_version: str

    def to_canonical_string(self) -> str:
        # deterministic, field-ordered, colon-separated
        items = []
        for f in sorted(fields(self), key=lambda x: x.name):
            items.append(f"{f.name}:{getattr(self, f.name)}")
        return "|".join(items)

@dataclass(frozen=True)
class KVFingerprint:
    model_id: str
    llm_endpoint_hash: str
    prompt_template_hash: str
    dataset_version: str

    def to_canonical_string(self) -> str:
        items = []
        for f in sorted(fields(self), key=lambda x: x.name):
            items.append(f"{f.name}:{getattr(self, f.name)}")
        return "|".join(items)
