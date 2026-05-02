from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Path, Response, status

from ..auth import verify_token
from ..db import pool

router = APIRouter()
log = logging.getLogger(__name__)


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_token)],
)
async def delete_session(session_id: str = Path(..., min_length=1, max_length=256)) -> Response:
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            # session-scoped memories
            await cur.execute(
                "DELETE FROM memory WHERE session_scope = %s", (session_id,)
            )
            # turns (cascades delete to memories sourced from these turns)
            await cur.execute(
                "DELETE FROM episodic_turn WHERE session_id = %s", (session_id,)
            )
            await conn.commit()
    log.info("deleted session=%s", session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
