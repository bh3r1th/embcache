import asyncio
import time
import os
from typing import Dict, Any, Optional
import httpx

from ._config import LLMConfig
from ._metrics import MetricsCollector, get_logger

_log = get_logger(__name__)

class LLMClient:
    def __init__(self, config: LLMConfig, metrics: MetricsCollector):
        self._config = config
        self._metrics = metrics
        self._client = httpx.AsyncClient(timeout=config.timeout_seconds)

    async def generate_kv_state(self, document: str) -> bytes:
        t0 = time.monotonic()
        try:
            api_key = self._config.api_key()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self._config.model_id,
                "messages": [{"role": "user", "content": document}],
                "max_tokens": self._config.max_tokens if self._config.max_tokens > 0 else None
            }
            # Strip None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            url = f"{self._config.endpoint.rstrip('/')}/chat/completions"
            response = await self._client.post(url, headers=headers, json=payload)
            
            if response.is_error:
                raise RuntimeError(f"LLM call failed: {response.status_code} {response.text[:200]}")
            
            content = response.json()["choices"][0]["message"]["content"]
            result_bytes = content.encode("utf-8")
            
            self._metrics.observe_kv_generation(time.monotonic() - t0)
            self._metrics.observe_kv_state_size(len(result_bytes))
            
            return result_bytes
        except Exception as e:
            _log.error(f"Error in generate_kv_state: {type(e).__name__}: {str(e)}")
            raise

    async def close(self) -> None:
        await self._client.aclose()
