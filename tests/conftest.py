"""Shared pytest fixtures.

The contract tests in this suite hit a *running* memory-service. The
expected workflow is::

    docker compose up -d
    until curl -sf http://localhost:8080/health; do sleep 1; done
    pytest

You can override the target with ``MEMORY_SERVICE_URL``.
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest


BASE_URL = os.environ.get("MEMORY_SERVICE_URL", "http://localhost:8080")


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session")
def client(base_url: str) -> httpx.Client:
    headers: dict[str, str] = {}
    token = os.environ.get("MEMORY_AUTH_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(base_url=base_url, headers=headers, timeout=60.0) as c:
        yield c


@pytest.fixture
def fresh_session() -> str:
    return f"sess-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def fresh_user(client: httpx.Client) -> str:
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    yield user_id
    # cleanup
    client.delete(f"/users/{user_id}")
