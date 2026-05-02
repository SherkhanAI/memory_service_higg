"""Shared httpx client for OpenRouter and direct-vendor calls.

OpenRouter is OpenAI-compatible for ``/chat/completions`` and
``/embeddings``, and exposes a Cohere-compatible ``/rerank``. We use a
single async client with a generous timeout (extraction LLM calls can be
slow under structured-output validation).
"""

from __future__ import annotations

import logging

import httpx

from ..config import settings

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _make_client() -> httpx.AsyncClient:
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter for telemetry / quota.
        "HTTP-Referer": "https://github.com/higgsfield-memory-service",
        "X-Title": "memory-service",
    }
    return httpx.AsyncClient(
        base_url=settings.openrouter_base_url,
        headers=headers,
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = _make_client()
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
