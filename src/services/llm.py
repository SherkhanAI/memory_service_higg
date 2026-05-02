"""OpenRouter chat completions wrapper with strict JSON Schema outputs.

OpenRouter is OpenAI-compatible at ``/chat/completions``. ``response_format``
with ``type=json_schema`` and ``strict=true`` constrains the model to a
syntactically valid JSON object matching the schema (provider-dependent;
``openai/gpt-5.4-mini`` supports it).
"""

from __future__ import annotations

import json
import logging
from typing import Any

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


async def chat_json(
    *,
    system: str,
    user: str,
    schema: dict[str, Any],
    schema_name: str = "response",
    temperature: float = 0.0,
    timeout_s: float = 45.0,
) -> dict | None:
    """Run a chat completion with strict JSON Schema. Returns parsed dict.

    Returns ``None`` on upstream failure or no content (callers should
    treat as 'no extraction'). Never raises — extraction failures must
    not crash ``/turns``.
    """
    if not settings.has_extraction_key:
        return None

    client = get_client()
    payload = {
        "model": settings.extraction_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        },
    }

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.6, min=0.5, max=4),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.RemoteProtocolError, httpx.ReadError)
            ),
            reraise=True,
        ):
            with attempt:
                r = await client.post(
                    "/chat/completions", json=payload, timeout=timeout_s
                )
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"upstream {r.status_code}", request=r.request, response=r
                    )
                r.raise_for_status()
                data = r.json()
                msg = data.get("choices", [{}])[0].get("message", {})
                content = msg.get("content") or ""
                if not content:
                    log.warning("chat_json: empty content; raw=%s", data)
                    return None
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    log.warning("chat_json: bad JSON: %s\n---\n%s", e, content[:500])
                    return None
    except httpx.HTTPStatusError as e:
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:
            pass
        log.warning("chat_json upstream error: %s body=%s", e, body)
        return None
    except Exception as e:  # pragma: no cover
        log.warning("chat_json failed: %s", e)
        return None

    return None
