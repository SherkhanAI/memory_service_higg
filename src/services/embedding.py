"""Embedding service.

Default provider is OpenRouter
(``POST https://openrouter.ai/api/v1/embeddings``, OpenAI-compatible).

Gemini Embedding 2 (preview) is trained with Matryoshka Representation
Learning, so we can request native 3072-d vectors and truncate-and-renormalize
to ``settings.embedding_dim`` (1536 by default to match the pgvector
column). We pass ``dimensions`` whenever we can — OpenRouter forwards the
parameter to the underlying provider when supported. We always re-truncate
on the client side as a safety net so the column shape is stable
regardless of upstream behaviour.

If no API key is configured, embed functions return ``None`` (or a list of
``None``s for batch). Callers must handle this — typically by writing the
turn without an embedding so contract tests keep passing.
"""

from __future__ import annotations

import logging
import math

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import settings
from .openrouter import get_client

log = logging.getLogger(__name__)


_INPUT_CHAR_CAP = 32_000  # ~8k tokens, matches gemini-embedding-2 context.


def _truncate_normalize(vec: list[float], target_dim: int) -> list[float]:
    """Matryoshka-truncate to target_dim and L2-renormalize."""
    if len(vec) == target_dim:
        return vec
    if len(vec) < target_dim:
        # pad with zeros and renormalize — vendor returned smaller than asked
        vec = vec + [0.0] * (target_dim - len(vec))
    else:
        vec = vec[:target_dim]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


async def _post_embed(payload: dict) -> dict:
    client = get_client()
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.RemoteProtocolError, httpx.ReadError)
        ),
        reraise=True,
    ):
        with attempt:
            r = await client.post("/embeddings", json=payload)
            if r.status_code >= 500:
                # Retry on upstream 5xx
                raise httpx.HTTPStatusError(
                    f"upstream {r.status_code}", request=r.request, response=r
                )
            r.raise_for_status()
            return r.json()
    raise RuntimeError("unreachable")


async def embed_one(text: str) -> list[float] | None:
    if not settings.has_embedding_key:
        return None
    text = (text or "").strip()
    if not text:
        return None
    text = text[:_INPUT_CHAR_CAP]

    payload: dict = {"model": settings.embedding_model, "input": text}
    if settings.embedding_dim:
        payload["dimensions"] = settings.embedding_dim

    try:
        data = await _post_embed(payload)
    except httpx.HTTPStatusError as e:
        log.warning("embedding upstream error: %s body=%s", e, getattr(e.response, "text", ""))
        return None
    except Exception as e:  # pragma: no cover
        log.warning("embedding failed: %s", e)
        return None

    vec = data["data"][0]["embedding"]
    return _truncate_normalize(list(vec), settings.embedding_dim)


async def embed_batch(texts: list[str]) -> list[list[float] | None]:
    if not settings.has_embedding_key or not texts:
        return [None] * len(texts)

    cleaned = [(i, (t or "").strip()[:_INPUT_CHAR_CAP]) for i, t in enumerate(texts)]
    real_inputs = [(i, t) for i, t in cleaned if t]

    out: list[list[float] | None] = [None] * len(texts)
    if not real_inputs:
        return out

    payload: dict = {
        "model": settings.embedding_model,
        "input": [t for _, t in real_inputs],
    }
    if settings.embedding_dim:
        payload["dimensions"] = settings.embedding_dim

    try:
        data = await _post_embed(payload)
    except Exception as e:  # pragma: no cover
        log.warning("batch embedding failed: %s", e)
        return out

    # Order results by their `index` field (per OpenAI spec).
    sorted_data = sorted(data["data"], key=lambda x: int(x.get("index", 0)))
    for (orig_idx, _), entry in zip(real_inputs, sorted_data):
        out[orig_idx] = _truncate_normalize(
            list(entry["embedding"]), settings.embedding_dim
        )
    return out
