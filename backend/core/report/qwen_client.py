"""Qwen / DashScope OpenAI-compatible client utilities.

This module intentionally avoids storing any secrets. The API key is read from
`backend.config.settings.settings` (which itself prefers environment variables).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import settings


class QwenClientError(RuntimeError):
    pass


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


@retry(
    reraise=True,
    stop=stop_after_attempt(max(1, settings.dashscope_max_retries)),
    wait=wait_exponential(multiplier=0.8, min=0.8, max=6),
    retry=retry_if_exception_type((httpx.HTTPError, QwenClientError)),
)
async def chat_completions(
    *,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> str:
    """Call DashScope compatible-mode chat completions and return message content."""

    api_key = (settings.dashscope_api_key or "").strip()
    if not api_key:
        raise QwenClientError("DashScope API key is empty")

    base_url = _normalize_base_url(settings.dashscope_base_url)
    url = f"{base_url}/chat/completions"

    payload: Dict[str, Any] = {
        "model": model or settings.dashscope_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    timeout = httpx.Timeout(settings.dashscope_timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise QwenClientError(f"DashScope request failed: {e.response.text}") from e

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        raise QwenClientError(f"Unexpected response schema: {json.dumps(data, ensure_ascii=False)[:500]}") from e
