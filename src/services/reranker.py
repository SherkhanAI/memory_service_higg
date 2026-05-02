"""Cross-encoder reranker. Provider-agnostic.

Three providers wired:
  - ``jina`` (default): direct ``https://api.jina.ai/v1/rerank``,
    ``jina-reranker-v3``. SOTA on BEIR (61.94 nDCG-10), sharper score
    distribution than Cohere on personal-fact corpora.
  - ``openrouter``: OpenRouter ``/api/v1/rerank``, e.g.
    ``cohere/rerank-4-fast``. Cheaper unified billing, but Cohere's
    score distribution is compressed for short personal facts.
  - ``cohere``: direct Cohere ``/v2/rerank``. Same scores as
    OpenRouter Cohere; available if you'd rather skip the gateway.

All three share the same response shape:
  ``{"results": [{"index": int, "relevance_score": float}, ...]}``

Returns the input candidates re-ordered by relevance with
``rerank_score`` populated. When no provider is configured, falls back
to the input order (already RRF-fused, so reasonable degradation).
"""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import settings
from .openrouter import get_client as get_openrouter_client
from .retrieval import Candidate

log = logging.getLogger(__name__)


_DEFAULT_TOP_K = 10
_DOC_CHAR_CAP = 1500


def _route() -> tuple[str | None, str, str | None, dict] | None:
    """Returns (url, model, api_key, extra_headers) or None if unconfigured."""
    p = settings.reranker_provider
    if p == "jina":
        if not settings.jina_api_key:
            return None
        return (
            "https://api.jina.ai/v1/rerank",
            settings.reranker_model,
            settings.jina_api_key,
            {},
        )
    if p == "cohere":
        if not settings.cohere_api_key:
            return None
        return (
            "https://api.cohere.com/v2/rerank",
            settings.reranker_model,
            settings.cohere_api_key,
            {},
        )
    if p == "openrouter":
        if not settings.openrouter_api_key:
            return None
        return (
            f"{settings.openrouter_base_url}/rerank",
            settings.reranker_model,
            settings.openrouter_api_key,
            {
                "HTTP-Referer": "https://github.com/higgsfield-memory-service",
                "X-Title": "memory-service",
            },
        )
    return None


async def rerank(
    *,
    query: str,
    candidates: list[Candidate],
    top_k: int = _DEFAULT_TOP_K,
) -> list[Candidate]:
    if not candidates:
        return []

    route = _route()
    if route is None:
        # No reranker configured → trust RRF fusion order.
        return candidates[:top_k]
    url, model, key, extra_headers = route

    docs = [c.text[:_DOC_CHAR_CAP] for c in candidates]
    payload = {
        "model": model,
        "query": query,
        "documents": docs,
        "top_n": min(top_k, len(docs)),
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **extra_headers,
    }

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.RemoteProtocolError, httpx.ReadError)
            ),
            reraise=True,
        ):
            with attempt:
                async with httpx.AsyncClient(timeout=20.0) as c:
                    r = await c.post(url, headers=headers, json=payload)
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"upstream {r.status_code}", request=r.request, response=r
                    )
                r.raise_for_status()
                data = r.json()
                break
    except Exception as e:  # pragma: no cover
        log.warning("reranker (%s) failed (%s) — falling back to RRF order",
                    settings.reranker_provider, e)
        return candidates[:top_k]

    out: list[Candidate] = []
    for result in data.get("results", []):
        try:
            idx = int(result["index"])
            score = float(result.get("relevance_score") or result.get("score") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates):
            c = candidates[idx]
            c.rerank_score = score
            out.append(c)

    if not out:
        return candidates[:top_k]
    return out
