"""Concurrent-isolation tests (task.md §5 + §9).

Two guarantees:
  1. Concurrent /turns requests across DIFFERENT users do not leak facts
     between users (cross-user isolation).
  2. Two users with the same session_id do not collide - session_id
     scope is per-user, not global.

We hit a live server with a small thread pool; no async machinery
required (httpx.Client is sync but thread-safe enough for this use).
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pytest


pytestmark = pytest.mark.concurrent


def _ingest(client: httpx.Client, user_id: str, session_id: str,
            content: str, ts: str) -> int:
    r = client.post(
        "/turns",
        json={
            "session_id": session_id,
            "user_id": user_id,
            "messages": [{"role": "user", "content": content}],
            "timestamp": ts,
        },
        timeout=120.0,
    )
    return r.status_code


def test_concurrent_users_no_leak(client: httpx.Client) -> None:
    """Parallel ingest of distinct facts across users; recall stays scoped."""
    users = [
        (f"alice_{uuid.uuid4().hex[:6]}", "I love sourdough bread."),
        (f"bob_{uuid.uuid4().hex[:6]}", "I am a competitive triathlete."),
        (f"carol_{uuid.uuid4().hex[:6]}", "I drive a 1995 Mazda Miata."),
    ]
    ts = "2026-04-02T09:00:00Z"

    try:
        with ThreadPoolExecutor(max_workers=len(users)) as ex:
            futs = [
                ex.submit(_ingest, client, uid, f"sess_{uid}", content, ts)
                for uid, content in users
            ]
            statuses = [f.result(timeout=180) for f in as_completed(futs)]
            assert all(s == 201 for s in statuses), (
                f"ingest failed: {statuses}"
            )

        # Each user's memories must NOT mention the other users' facts.
        keywords = {
            users[0][0]: ["sourdough", "bread"],
            users[1][0]: ["triathlete", "triathlon"],
            users[2][0]: ["mazda", "miata"],
        }

        for uid, _ in users:
            mems = client.get(f"/users/{uid}/memories").json().get("memories", [])
            text_blob = " ".join(
                f"{m.get('key','')} {m.get('value','')}".lower() for m in mems
            )
            for other_uid, other_kws in keywords.items():
                if other_uid == uid:
                    continue
                for kw in other_kws:
                    assert kw not in text_blob, (
                        f"user {uid} memories leaked '{kw}' from {other_uid}; "
                        f"blob: {text_blob[:200]}"
                    )
    finally:
        for uid, _ in users:
            client.delete(f"/users/{uid}")


def test_shared_session_id_isolated_per_user(client: httpx.Client) -> None:
    """Two users using the literal same session_id must not see each other."""
    shared_session = f"shared_{uuid.uuid4().hex[:6]}"
    u1 = f"u1_{uuid.uuid4().hex[:6]}"
    u2 = f"u2_{uuid.uuid4().hex[:6]}"
    ts = "2026-04-02T10:00:00Z"

    try:
        s1 = _ingest(client, u1, shared_session, "I just adopted a cat named Whiskers.", ts)
        s2 = _ingest(client, u2, shared_session, "I just bought a kayak.", ts)
        assert s1 == 201 and s2 == 201

        r = client.post("/recall", json={
            "query": "What pet does the user have?",
            "session_id": shared_session,
            "user_id": u1,
            "max_tokens": 512,
        }).json()
        ctx = (r.get("context") or "").lower()
        assert "kayak" not in ctx, (
            f"u1 recall leaked u2's kayak fact under shared session: {ctx[:200]}"
        )

        r = client.post("/recall", json={
            "query": "What did the user buy recently?",
            "session_id": shared_session,
            "user_id": u2,
            "max_tokens": 512,
        }).json()
        ctx = (r.get("context") or "").lower()
        assert "whiskers" not in ctx and "cat" not in ctx, (
            f"u2 recall leaked u1's cat fact under shared session: {ctx[:200]}"
        )
    finally:
        client.delete(f"/users/{u1}")
        client.delete(f"/users/{u2}")
