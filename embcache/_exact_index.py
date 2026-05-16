from collections import OrderedDict
import numpy as np
from typing import Any
from ._metrics import MetricsCollector, get_logger

_log = get_logger(__name__)

class ExactIndex:
    def __init__(self, max_entries: int, metrics: MetricsCollector):
        self.max_entries = max_entries
        self.metrics = metrics
        self.lru = OrderedDict()
        if max_entries == 0:
            _log.warning("ExactIndex disabled (max_entries=0)")

    def get(self, key: str) -> np.ndarray | None:
        try:
            if self.max_entries == 0:
                return None
            if key in self.lru:
                self.lru.move_to_end(key)
                self.metrics.record_exact_hit()
                return np.copy(self.lru[key])
            return None
        except Exception as e:
            _log.error(f"Error in ExactIndex.get for {key}: {e}")
            return None

    def put(self, key: str, vector: np.ndarray) -> None:
        try:
            if self.max_entries == 0:
                return
            if key in self.lru:
                self.lru.pop(key)
            elif len(self.lru) >= self.max_entries:
                self.lru.popitem(last=False)
            
            self.lru[key] = np.copy(vector)
        except Exception as e:
            _log.error(f"Error in ExactIndex.put for {key}: {e}")

    def invalidate(self, key: str) -> bool:
        try:
            if key in self.lru:
                self.lru.pop(key)
                return True
            return False
        except Exception:
            return False

    def __len__(self) -> int:
        return len(self.lru)

    def stats(self) -> dict:
        return {
            "entry_count": len(self.lru),
            "max_entries": self.max_entries
        }
