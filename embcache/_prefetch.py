import asyncio
import collections
from typing import Callable, Awaitable, List, Dict, Set, Optional
from ._metrics import get_logger

_log = get_logger(__name__)

class PrefetchEngine:
    def __init__(
        self,
        fetch_fn: Callable[[str], Awaitable[List[float]]],
        enabled: bool = False,
    ):
        self._fetch_fn = fetch_fn
        self._enabled = enabled
        self._cooccurrence: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self._prefetch_hits: int = 0
        self._prefetch_total: int = 0
        self._polluted: int = 0
        self._pending: Set[str] = set()

    def record_access(self, key: str, context_keys: List[str]) -> None:
        if not self._enabled:
            return
        for ctx_key in context_keys:
            self._cooccurrence[ctx_key][key] += 1

    async def maybe_prefetch(
        self,
        current_key: str,
        cache_put_fn: Callable[[str, List[float]], Awaitable[None]],
        top_k: int = 3,
    ) -> None:
        if not self._enabled:
            return
            
        candidates = [k for k, _ in self._cooccurrence[current_key].most_common(top_k)]
        for candidate in candidates:
            if candidate not in self._pending:
                self._pending.add(candidate)
                asyncio.create_task(self._do_prefetch(candidate, cache_put_fn))

    async def _do_prefetch(
        self,
        key: str,
        cache_put_fn: Callable[[str, List[float]], Awaitable[None]],
    ) -> None:
        try:
            vector = await self._fetch_fn(key)
            await cache_put_fn(key, vector)
            self._prefetch_total += 1
        except Exception as e:
            _log.warning(f"Prefetch failed for {key}: {e}")
        finally:
            self._pending.discard(key)

    def record_prefetch_hit(self, key: str) -> None:
        if self._enabled:
            self._prefetch_hits += 1

    def hit_rate(self) -> float:
        if self._prefetch_total == 0:
            return 0.0
        return self._prefetch_hits / self._prefetch_total

    def pollution_rate(self) -> float:
        if self._prefetch_total == 0:
            return 0.0
        return (self._prefetch_total - self._prefetch_hits) / self._prefetch_total

    def stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "prefetch_hits": self._prefetch_hits,
            "prefetch_total": self._prefetch_total,
            "hit_rate": self.hit_rate(),
            "pollution_rate": self.pollution_rate(),
            "pending_count": len(self._pending),
        }
