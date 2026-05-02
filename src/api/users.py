from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Path, Response, status

from ..auth import verify_token
from ..db import pool
from ..schemas import MemoriesOut, MemoryRecord

router = APIRouter()
log = logging.getLogger(__name__)


_CONFIDENCE_NUM = {"low": 0.3, "med": 0.6, "high": 0.9}


@router.get(
    "/users/{user_id}/memories",
    response_model=MemoriesOut,
    dependencies=[Depends(verify_token)],
)
async def list_memories(user_id: str = Path(..., min_length=1, max_length=256)) -> MemoriesOut:
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT m.id::text,
                       m.kind::text,
                       m.predicate,
                       m.object_text,
                       m.confidence::text,
                       m.session_scope,
                       m.source_turn_id::text,
                       m.t_created,
                       COALESCE(m.t_invalid, m.t_created) AS updated_at,
                       (SELECT s.id::text FROM memory s
                          WHERE s.superseded_by = m.id LIMIT 1) AS supersedes,
                       (m.t_invalid IS NULL) AS active
                FROM memory m
                WHERE m.user_id = %s
                ORDER BY m.t_created DESC
                """,
                (user_id,),
            )
            rows = await cur.fetchall()

    memories = [
        MemoryRecord(
            id=r[0],
            type=r[1],
            key=r[2],
            value=r[3],
            confidence=_CONFIDENCE_NUM.get(r[4], 0.5),
            source_session=r[5],
            source_turn=r[6],
            created_at=r[7],
            updated_at=r[8],
            supersedes=r[9],
            active=r[10],
        )
        for r in rows
    ]
    return MemoriesOut(memories=memories)


@router.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_token)],
)
async def delete_user(user_id: str = Path(..., min_length=1, max_length=256)) -> Response:
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            # entity & memory_entity cascade off memory; memory has no FK to user
            await cur.execute("DELETE FROM memory WHERE user_id = %s", (user_id,))
            await cur.execute("DELETE FROM entity WHERE user_id = %s", (user_id,))
            await cur.execute(
                "DELETE FROM episodic_turn WHERE user_id = %s", (user_id,)
            )
            await conn.commit()
    log.info("deleted user=%s", user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
