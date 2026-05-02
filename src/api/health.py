from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..db import pool

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    try:
        async with pool().connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                await cur.fetchone()
    except Exception as exc:  # pragma: no cover - failure path
        raise HTTPException(status_code=503, detail=f"db not ready: {exc}") from exc
    return {"status": "ok"}
