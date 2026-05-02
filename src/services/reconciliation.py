"""Mem0-style reconciliation with bi-temporal write semantics.

Pipeline:
    extracted facts → group by predicate → retrieve top-K active rows
    per predicate → LLM judges ADD/UPDATE/SUPERSEDE/NOOP per fact →
    apply: insert / merge / invalidate-old-and-insert-new.

Bi-temporal model: an UPDATE never destructively overwrites — it
either bumps mention_count, merges qualifiers, or stamps the old row
with t_invalid (and links via superseded_by) and writes a new row
with the current t_valid. Inspectable via /users/{id}/memories.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from pgvector.psycopg import Vector

from ..db import pool
from . import predicates
from .embedding import embed_one
from .llm import chat_json

log = logging.getLogger(__name__)


_RECONCILE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "fact_index": {"type": "integer"},
                    "action": {"enum": ["ADD", "UPDATE", "SUPERSEDE", "NOOP"]},
                    "target_id": {"type": ["string", "null"]},
                    "reasoning": {"type": "string"},
                },
                "required": ["fact_index", "action", "target_id", "reasoning"],
            },
        },
    },
    "required": ["decisions"],
}


# ---- DB helpers --------------------------------------------------------------


async def _retrieve_existing(user_id: str, predicate: str, k: int = 5) -> list[dict]:
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id::text, predicate, object_text, kind, stance::text,
                       confidence::text, t_valid, t_created
                FROM memory
                WHERE user_id = %s AND predicate = %s AND t_invalid IS NULL
                ORDER BY t_created DESC
                LIMIT %s
                """,
                (user_id, predicate, k),
            )
            rows = await cur.fetchall()
    return [
        {
            "id": r[0],
            "predicate": r[1],
            "object_text": r[2],
            "kind": r[3],
            "stance": r[4],
            "confidence": r[5],
            "t_valid": r[6].isoformat() if r[6] else None,
            "t_created": r[7].isoformat() if r[7] else None,
        }
        for r in rows
    ]


async def _insert_memory(
    user_id: str, fact: dict, source_turn_id: str, ts: datetime
) -> str:
    text = _fact_text(fact)
    embedding = await embed_one(text)
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO memory
                    (user_id, predicate, object_text, object_qualifiers,
                     kind, stance, confidence, is_implicit,
                     t_valid, source_turn_id, source_text,
                     embedding, tsv)
                VALUES (%s, %s, %s, %s::jsonb,
                        %s::memory_kind, %s, %s::memory_confidence, %s,
                        %s, %s::uuid, %s,
                        %s, to_tsvector('english', %s))
                RETURNING id
                """,
                (
                    user_id,
                    fact["predicate"],
                    fact["object_text"],
                    "{}",
                    fact["kind"],
                    fact.get("stance"),
                    fact.get("confidence", "med"),
                    bool(fact.get("is_implicit", False)),
                    ts,
                    source_turn_id,
                    fact.get("source_text", "")[:2000],
                    Vector(embedding) if embedding is not None else None,
                    text,
                ),
            )
            row = await cur.fetchone()
            await conn.commit()
    return str(row[0])


async def _invalidate(memory_id: str, ts: datetime) -> None:
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE memory
                SET t_invalid = %s, t_expired = %s
                WHERE id = %s::uuid AND t_invalid IS NULL
                """,
                (ts, ts, memory_id),
            )
            await conn.commit()


async def _link_supersession(old_id: str, new_id: str) -> None:
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE memory SET superseded_by = %s::uuid WHERE id = %s::uuid",
                (new_id, old_id),
            )
            await conn.commit()


async def _bump_mention(memory_id: str) -> None:
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE memory SET mention_count = mention_count + 1 WHERE id = %s::uuid",
                (memory_id,),
            )
            await conn.commit()


# ---- LLM judge ---------------------------------------------------------------


def _reconcile_system_prompt() -> str:
    excl = ", ".join(sorted(predicates.EXCLUSIVE_PREDICATES))
    multi = ", ".join(sorted(predicates.MULTI_VALUED_PREDICATES))
    return f"""You are reconciling NEW facts against EXISTING active facts in a user's memory store.

For each new fact, choose ONE action:
- ADD: new fact, unrelated to anything existing — write it as a fresh row.
- UPDATE: complementary detail (e.g. mentions same employer at same job, with extra context) — merge into the existing row.
- SUPERSEDE: contradicts an existing active row — mark old invalid, write new. Use this for state changes (changed jobs, moved cities, opinion shift).
- NOOP: same as existing (duplicate or paraphrase) — no new row, just bump counter.

Hard rules:
- EXCLUSIVE predicates ({excl}): if same predicate exists with a DIFFERENT object_text → SUPERSEDE. Same object_text → NOOP.
- MULTI_VALUED predicates ({multi}): different object_text → ADD. Same object_text → NOOP.
- opinion_about: if same object_text and stance differs (e.g. positive→negative) → SUPERSEDE; same stance → NOOP; new stance qualifier shifts ("love TS" → "TS for big projects, Python for scripts") → SUPERSEDE.
- "other:*" predicates: ADD unless an existing matches in object_text → NOOP.

Output:
- For SUPERSEDE / UPDATE / NOOP, set target_id to the EXISTING memory id you're acting on.
- For ADD, target_id MUST be null.
- Always include a one-sentence reasoning."""


async def _judge(
    new_facts: list[dict], existing_per_predicate: dict[str, list[dict]]
) -> list[dict]:
    if not new_facts:
        return []

    lines = ["NEW FACTS:"]
    for i, f in enumerate(new_facts):
        lines.append(
            f"  [{i}] predicate={f['predicate']} "
            f"object={f['object_text']!r} "
            f"kind={f['kind']} "
            f"stance={f.get('stance')} "
            f"source={f.get('source_text', '')[:120]!r}"
        )
    lines.append("\nEXISTING ACTIVE FACTS:")
    if not any(existing_per_predicate.values()):
        lines.append("  (none)")
    else:
        for pred, rows in existing_per_predicate.items():
            for r in rows:
                lines.append(
                    f"  id={r['id']} predicate={pred} "
                    f"object={r['object_text']!r} "
                    f"kind={r['kind']} stance={r.get('stance')} "
                    f"t_valid={r['t_valid']}"
                )

    user = "\n".join(lines) + "\n\nReturn one decision per new fact (by fact_index)."

    result = await chat_json(
        system=_reconcile_system_prompt(),
        user=user,
        schema=_RECONCILE_SCHEMA,
        schema_name="reconciliation",
        timeout_s=30.0,
    )
    if not result or "decisions" not in result:
        log.warning("reconcile: LLM judge failed; falling back to ADD-all")
        return [
            {"fact_index": i, "action": "ADD", "target_id": None, "reasoning": "fallback"}
            for i in range(len(new_facts))
        ]
    return result["decisions"] or []


# ---- Public entry ------------------------------------------------------------


async def reconcile_and_write(
    *,
    user_id: str | None,
    facts: list[dict],
    source_turn_id: str,
    turn_ts: datetime,
) -> dict[str, int]:
    """Returns a counts dict: {add, update, supersede, noop}."""
    counts = {"add": 0, "update": 0, "supersede": 0, "noop": 0}
    if not user_id or not facts:
        return counts

    # Group existing rows by predicate (only those we'll consider).
    existing: dict[str, list[dict]] = {}
    for pred in sorted({f["predicate"] for f in facts}):
        rows = await _retrieve_existing(user_id, pred, k=5)
        if rows:
            existing[pred] = rows

    decisions = await _judge(facts, existing)
    by_index = {int(d["fact_index"]): d for d in decisions if "fact_index" in d}

    for i, fact in enumerate(facts):
        d = by_index.get(i, {"action": "ADD", "target_id": None})
        action = d.get("action", "ADD")
        target_id = d.get("target_id")

        try:
            if action == "NOOP":
                if target_id:
                    await _bump_mention(target_id)
                counts["noop"] += 1

            elif action == "UPDATE":
                if target_id:
                    await _bump_mention(target_id)  # qualifiers merge intentionally minimal
                    counts["update"] += 1
                else:
                    await _insert_memory(user_id, fact, source_turn_id, turn_ts)
                    counts["add"] += 1

            elif action == "SUPERSEDE":
                if target_id:
                    await _invalidate(target_id, turn_ts)
                new_id = await _insert_memory(user_id, fact, source_turn_id, turn_ts)
                if target_id:
                    await _link_supersession(target_id, new_id)
                counts["supersede"] += 1

            else:  # ADD or unknown
                await _insert_memory(user_id, fact, source_turn_id, turn_ts)
                counts["add"] += 1

        except Exception as e:  # pragma: no cover
            log.exception("reconcile apply failed for fact[%d] action=%s: %s",
                          i, action, e)

    return counts


def _fact_text(fact: dict[str, Any]) -> str:
    """Render fact for embedding/tsv. Stable across actions."""
    parts = [fact["predicate"], fact["object_text"]]
    if fact.get("stance"):
        parts.append(f"({fact['stance']})")
    if src := fact.get("source_text"):
        parts.append(src)
    return " ".join(p for p in parts if p)
