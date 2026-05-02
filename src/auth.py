from __future__ import annotations

from fastapi import Header, HTTPException, status

from .config import settings


async def verify_token(authorization: str | None = Header(default=None)) -> None:
    """No-op when MEMORY_AUTH_TOKEN is empty.

    Otherwise enforce ``Authorization: Bearer <token>`` exact match.
    """
    expected = settings.memory_auth_token
    if not expected:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    presented = authorization.split(" ", 1)[1].strip()
    if presented != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
