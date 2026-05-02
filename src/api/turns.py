from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, status
from pgvector.psycopg import Vector

from ..auth import verify_token
from ..config import settings
from ..db import pool
from ..schemas import TurnIn, TurnOut
from ..services.embedding import embed_one
from ..services.extraction import extract_facts
from ..services.reconciliation import reconcile_and_write

router = APIRouter()
log = logging.getLogger(__name__)


def _flatten_messages(turn: TurnIn) -> str:
    parts: list[str] = []
    for m in turn.messages:
        prefix = m.role.upper()
        if m.role == "tool" and m.name:
            prefix = f"TOOL:{m.name}"
        parts.append(f"{prefix}: {m.content}")
    return "\n".join(parts)


async def _recent_session_text(session_id: str, exclude_id: str, limit: int = 2) -> str | None:
    """Fetch up to N most recent prior turns in this session for coreference context."""
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT raw_text
                FROM episodic_turn
                WHERE session_id = %s AND id <> %s::uuid
                ORDER BY ts DESC, created_at DESC
                LIMIT %s
                """,
                (session_id, exclude_id, limit),
            )
            rows = await cur.fetchall()
    if not rows:
        return None
    # Reverse to chronological
    return "\n\n".join(r[0] for r in reversed(rows) if r[0])


@router.post(
    "/turns",
    response_model=TurnOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_token)],
)
async def post_turn(turn: TurnIn) -> TurnOut:
    raw_text = _flatten_messages(turn)
    messages_json = json.dumps([m.model_dump() for m in turn.messages])
    metadata_json = json.dumps(turn.metadata)

    # 1. Embed the raw turn (synchronous — /turns blocks).
    embedding = await embed_one(raw_text)
    if embedding is not None and len(embedding) != settings.embedding_dim:
        log.warning(
            "embedding dim mismatch: got %d, expected %d — dropping",
            len(embedding), settings.embedding_dim,
        )
        embedding = None

    # 2. Insert episodic_turn first so source_turn_id FK is satisfied.
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO episodic_turn
                    (session_id, user_id, messages, ts, metadata,
                     raw_text, embedding, tsv)
                VALUES (%s, %s, %s::jsonb, %s, %s::jsonb,
                        %s, %s, to_tsvector('english', %s))
                RETURNING id
                """,
                (
                    turn.session_id,
                    turn.user_id,
                    messages_json,
                    turn.timestamp,
                    metadata_json,
                    raw_text,
                    Vector(embedding) if embedding is not None else None,
                    raw_text,
                ),
            )
            row = await cur.fetchone()
            await conn.commit()
    turn_id = str(row[0])

    # 3. Extract facts and reconcile (synchronous per ТЗ).
    #    Failures here MUST NOT crash /turns — log and continue.
    extracted = 0
    counts: dict[str, int] = {}
    if turn.user_id and settings.has_extraction_key:
        try:
            session_context = await _recent_session_text(turn.session_id, exclude_id=turn_id, limit=2)
            facts = await extract_facts(
                raw_text, turn.timestamp, session_context=session_context,
            )
            extracted = len(facts)
            counts = await reconcile_and_write(
                user_id=turn.user_id,
                facts=facts,
                source_turn_id=turn_id,
                turn_ts=turn.timestamp,
            )
        except Exception as e:  # pragma: no cover
            log.exception("extraction pipeline failed for turn=%s: %s", turn_id, e)

    log.info(
        "turn ingested session=%s user=%s id=%s msgs=%d embedded=%s extracted=%d "
        "add=%d update=%d supersede=%d noop=%d",
        turn.session_id, turn.user_id, turn_id, len(turn.messages),
        embedding is not None, extracted,
        counts.get("add", 0), counts.get("update", 0),
        counts.get("supersede", 0), counts.get("noop", 0),
    )
    return TurnOut(id=turn_id)
