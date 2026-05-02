"""v0.1 contract roundtrip & shape tests.

These cover only what v0.1 promises:
  - all 7 endpoints respond with the documented shape and status code
  - malformed input gets 4xx, never 5xx
  - the service is alive and persisting raw turns

Recall quality, supersession, and multi-hop tests live in test_memeval.py
once those features land in v0.3+.
"""

from __future__ import annotations

import httpx
import pytest


pytestmark = pytest.mark.contract


def test_health(client: httpx.Client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"


def test_turns_roundtrip(client: httpx.Client, fresh_session: str, fresh_user: str) -> None:
    payload = {
        "session_id": fresh_session,
        "user_id": fresh_user,
        "messages": [
            {"role": "user", "content": "I just moved to Berlin from NYC last month."},
            {"role": "assistant", "content": "How are you settling in?"},
        ],
        "timestamp": "2026-05-01T10:30:00Z",
        "metadata": {},
    }
    r = client.post("/turns", json=payload)
    assert r.status_code == 201, r.text
    body = r.json()
    assert isinstance(body.get("id"), str) and len(body["id"]) > 0


def test_recall_cold_session_is_empty_not_error(client: httpx.Client) -> None:
    r = client.post(
        "/recall",
        json={
            "query": "What does the user do for fun?",
            "session_id": "never-seen",
            "user_id": "ghost-user",
            "max_tokens": 256,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "context" in body and "citations" in body
    assert isinstance(body["citations"], list)


def test_search_shape(client: httpx.Client) -> None:
    r = client.post(
        "/search",
        json={"query": "anything", "user_id": "ghost", "session_id": None, "limit": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert "results" in body and isinstance(body["results"], list)


def test_user_memories_shape(client: httpx.Client, fresh_user: str) -> None:
    r = client.get(f"/users/{fresh_user}/memories")
    assert r.status_code == 200
    body = r.json()
    assert "memories" in body and isinstance(body["memories"], list)


def test_delete_session_idempotent(client: httpx.Client) -> None:
    r = client.delete("/sessions/never-existed")
    assert r.status_code == 204


def test_delete_user_idempotent(client: httpx.Client) -> None:
    r = client.delete("/users/never-existed")
    assert r.status_code == 204


# --- Resilience: 4xx, not 5xx ---


def test_malformed_json_is_400(client: httpx.Client) -> None:
    r = client.post(
        "/turns",
        content="not-json",
        headers={"Content-Type": "application/json"},
    )
    assert 400 <= r.status_code < 500


def test_missing_required_field_is_422(client: httpx.Client) -> None:
    r = client.post(
        "/turns",
        json={"session_id": "x"},  # missing messages, timestamp
    )
    assert r.status_code == 422


def test_unicode_safe(client: httpx.Client, fresh_session: str, fresh_user: str) -> None:
    payload = {
        "session_id": fresh_session,
        "user_id": fresh_user,
        "messages": [
            {"role": "user", "content": "Меня зовут Шерхан 🐯, люблю 寿司 and crème brûlée."},
        ],
        "timestamp": "2026-05-01T10:30:00Z",
        "metadata": {"locale": "ru-RU"},
    }
    r = client.post("/turns", json=payload)
    assert r.status_code == 201, r.text


def test_oversized_payload_does_not_crash(
    client: httpx.Client, fresh_session: str, fresh_user: str
) -> None:
    payload = {
        "session_id": fresh_session,
        "user_id": fresh_user,
        "messages": [
            {"role": "user", "content": "x" * 200_000},
        ],
        "timestamp": "2026-05-01T10:30:00Z",
        "metadata": {},
    }
    r = client.post("/turns", json=payload)
    # 201 (we accept) or 413 (we reject) are both fine; 5xx is not.
    assert r.status_code < 500
