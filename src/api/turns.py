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
from ..services.extraction import build_contextual_prefix, extract_facts
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


async def _recent_session_text_pre(session_id: str, limit: int = 2) -> str | None:
    """Fetch up to N most recent prior turns in this session.

    Called BEFORE the current turn is inserted, so no exclusion needed.
    Used both for the contextual prefix and for extraction coreference.
    """
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT raw_text
                FROM episodic_turn
                WHERE session_id = %s
                ORDER BY ts DESC, created_at DESC
                LIMIT %s
                """,
                (session_id, limit),
            )
            rows = await cur.fetchall()
    if not rows:
        return None
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

    # 1. Build an Anthropic-style contextual prefix using prior turns of
    #    this session. The prefix is prepended to raw_text for embedding
    #    and BM25 indexing (so retrieval keys on it), but raw_text stays
    #    clean for display in /recall context. Best-effort: failure here
    #    falls back to the bare raw_text.
    prior_session_text = (
        await _recent_session_text_pre(turn.session_id, limit=2)
        if turn.user_id else None
    )
    context_prefix: str | None = None
    if turn.user_id and settings.has_extraction_key:
        try:
            context_prefix = await build_contextual_prefix(
                raw_text,
                session_context=prior_session_text,
                session_id=turn.session_id,
                turn_ts=turn.timestamp,
            )
        except Exception as e:  # pragma: no cover
            log.warning("contextual prefix failed: %s", e)
            context_prefix = None

    # 2. Embed the (prefix + raw) text. /turns blocks until done per spec.
    text_for_index = (
        f"{context_prefix}\n\n{raw_text}" if context_prefix else raw_text
    )
    embedding = await embed_one(text_for_index)
    if embedding is not None and len(embedding) != settings.embedding_dim:
        log.warning(
            "embedding dim mismatch: got %d, expected %d - dropping",
            len(embedding), settings.embedding_dim,
        )
        embedding = None

    # 3. Insert episodic_turn first so source_turn_id FK is satisfied.
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO episodic_turn
                    (session_id, user_id, messages, ts, metadata,
                     raw_text, context_prefix, embedding, tsv)
                VALUES (%s, %s, %s::jsonb, %s, %s::jsonb,
                        %s, %s, %s, to_tsvector('english', %s))
                RETURNING id
                """,
                (
                    turn.session_id,
                    turn.user_id,
                    messages_json,
                    turn.timestamp,
                    metadata_json,
                    raw_text,
                    context_prefix,
                    Vector(embedding) if embedding is not None else None,
                    text_for_index,
                ),
            )
            row = await cur.fetchone()
            await conn.commit()
    turn_id = str(row[0])

    # 4. Extract facts and reconcile (synchronous per ТЗ).
    #    Failures here MUST NOT crash /turns - log and continue.
    extracted = 0
    counts: dict[str, int] = {}
    if turn.user_id and settings.has_extraction_key:
        try:
            facts = await extract_facts(
                raw_text, turn.timestamp,
                session_context=prior_session_text,
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
