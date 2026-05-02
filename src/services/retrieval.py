"""Hybrid retrieval over memory + episodic_turn.

Four parallel streams:
    1. memory   dense  — cosine on memory.embedding (t_invalid IS NULL)
    2. memory   sparse — ts_rank_cd on memory.tsv (BM25-flavoured)
    3. episodic dense  — cosine on episodic_turn.embedding
    4. episodic sparse — ts_rank_cd on episodic_turn.tsv

Fused via Reciprocal Rank Fusion (RRF) with k=60 (Cormack et al. 2009).
Output is a unified ``list[Candidate]`` ranked by fusion score, ready for
the cross-encoder reranker.

Why RRF: rank-based fusion is robust to score-scale differences between
sparse and dense (one is BM25 magnitudes, the other ∈ [0,1] cosine).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pgvector.psycopg import Vector

from ..db import pool
from .embedding import embed_one


_RRF_K = 60
_K_PER_STREAM = 30


@dataclass
class Candidate:
    key: str           # f"{source}:{id}" — unique per row across tables
    id: str            # uuid string
    source: str        # 'memory' | 'episodic'
    text: str          # rendered for reranker
    snippet: str       # rendered for /recall citations
    fields: dict[str, Any] = field(default_factory=dict)
    fusion_score: float = 0.0
    raw_dense_score: float | None = None  # cosine similarity of top-1 stream
    rerank_score: float | None = None


@dataclass
class HybridResult:
    candidates: list[Candidate]
    memory_dense_top1: float = 0.0
    episodic_dense_top1: float = 0.0


# Natural-language verbalization templates per canonical predicate.
# Cross-encoder rerankers (Jina v3 in particular) score document
# probability — they were trained on full sentences, not key:value
# triplets. Verbalising "employer: Notion" → "The user currently works
# at Notion." is the single biggest precision lever in this pipeline.
_VERBALIZATIONS: dict[str, str] = {
    "name":               "The user's name is {o}.",
    "age":                "The user is {o} years old.",
    "gender":             "The user's gender is {o}.",
    "lives_in":           "The user lives in {o}.",
    "lived_in":           "The user previously lived in {o}.",
    "born_in":            "The user was born in {o}.",
    "nationality":        "The user is {o}.",
    "employer":           "The user currently works at {o}.",
    "role":               "The user's job title is {o}.",
    "employer_past":      "The user previously worked at {o}.",
    "employer_start":     "The user started at the current job in {o}.",
    "degree":             "The user has the degree {o}.",
    "school":             "The user attended {o}.",
    "field_of_study":     "The user studied {o}.",
    "spouse":             "The user's spouse is {o}.",
    "partner":            "The user's partner is {o}.",
    "family.parent":      "The user's parent is {o}.",
    "family.child":       "The user has a child named {o}.",
    "family.sibling":     "The user has a sibling named {o}.",
    "friend":             "The user has a friend named {o}.",
    "pet":                "The user has a pet named {o}.",
    "preference.food":    "The user's food preference: {o}.",
    "preference.language":"The user uses the language {o}.",
    "preference.tool":    "The user prefers the tool {o}.",
    "preference.framework":"The user prefers the framework {o}.",
    "preference.activity":"The user enjoys {o}.",
    "opinion_about":      "The user has an opinion about {o}.",
    "event.travel":       "The user travelled to {o}.",
    "event.health":       "The user had a health event: {o}.",
    "event.life_change":  "A major life change for the user: {o}.",
    "skill":              "The user has the skill {o}.",
    "hobby":              "The user's hobby is {o}.",
}


def _verbalize(predicate: str, object_text: str, stance: str | None = None) -> str:
    template = _VERBALIZATIONS.get(predicate)
    if not template:
        # Fallback for "other:*" or unknown predicates
        clean = predicate.replace("other:", "").replace("_", " ")
        return f"The user's {clean} is {object_text}."
    out = template.format(o=object_text)
    if predicate == "opinion_about" and stance and stance != "none":
        out = f"The user feels {stance} about {object_text}."
    return out


def _make_memory_text(
    predicate: str,
    object_text: str,
    source_text: str | None,
    stance: str | None = None,
) -> str:
    sentence = _verbalize(predicate, object_text, stance)
    if source_text:
        return f'{sentence} They said: "{source_text.strip()[:200]}"'
    return sentence


def _make_episodic_text(raw_text: str) -> str:
    return (raw_text or "").strip()[:1200]


async def _memory_dense(user_id: str, q_vec: Vector, k: int) -> list[tuple]:
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id::text, predicate, object_text, stance, confidence::text,
                       is_implicit, source_text, t_valid, source_turn_id::text,
                       1 - (embedding <=> %s) AS score
                FROM memory
                WHERE user_id = %s
                  AND t_invalid IS NULL
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (q_vec, user_id, q_vec, k),
            )
            return await cur.fetchall()


async def _memory_sparse(user_id: str, query: str, k: int) -> list[tuple]:
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id::text, predicate, object_text, stance, confidence::text,
                       is_implicit, source_text, t_valid, source_turn_id::text,
                       ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS score
                FROM memory
                WHERE user_id = %s
                  AND t_invalid IS NULL
                  AND tsv @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, user_id, query, k),
            )
            return await cur.fetchall()


async def _episodic_dense(
    user_id: str | None,
    q_vec: Vector,
    k: int,
    session_id: str | None = None,
) -> list[tuple]:
    where = []
    args: list[Any] = [q_vec]
    if user_id:
        where.append("user_id = %s")
        args.append(user_id)
    if session_id:
        where.append("session_id = %s")
        args.append(session_id)
    where.append("embedding IS NOT NULL")
    sql = f"""
        SELECT id::text, raw_text, ts, session_id,
               1 - (embedding <=> %s) AS score
        FROM episodic_turn
        WHERE {' AND '.join(where)}
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    args.extend([q_vec, k])
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, tuple(args))
            return await cur.fetchall()


async def _episodic_sparse(
    user_id: str | None,
    query: str,
    k: int,
    session_id: str | None = None,
) -> list[tuple]:
    where = ["tsv @@ plainto_tsquery('english', %s)"]
    args: list[Any] = [query, query]
    if user_id:
        where.append("user_id = %s")
        args.append(user_id)
    if session_id:
        where.append("session_id = %s")
        args.append(session_id)
    sql = f"""
        SELECT id::text, raw_text, ts, session_id,
               ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS score
        FROM episodic_turn
        WHERE {' AND '.join(where)}
        ORDER BY score DESC
        LIMIT %s
    """
    args.append(k)
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, tuple(args))
            return await cur.fetchall()


def _candidate_from_memory(row: tuple) -> Candidate:
    (mid, predicate, object_text, stance, confidence, is_implicit,
     source_text, t_valid, source_turn_id, _score) = row
    text = _make_memory_text(predicate, object_text, source_text, stance)
    snippet = f"{predicate}: {object_text}"
    return Candidate(
        key=f"memory:{mid}",
        id=mid,
        source="memory",
        text=text,
        snippet=snippet,
        fields={
            "predicate": predicate,
            "object_text": object_text,
            "stance": stance,
            "confidence": confidence,
            "is_implicit": bool(is_implicit),
            "source_text": source_text,
            "t_valid": t_valid,
            "source_turn_id": source_turn_id,
        },
    )


def _candidate_from_episodic(row: tuple) -> Candidate:
    (eid, raw_text, ts, session_id, _score) = row
    text = _make_episodic_text(raw_text)
    snippet = (raw_text or "").strip().replace("\n", " ")
    if len(snippet) > 240:
        snippet = snippet[:240] + "..."
    return Candidate(
        key=f"episodic:{eid}",
        id=eid,
        source="episodic",
        text=text,
        snippet=snippet,
        fields={"raw_text": raw_text, "ts": ts, "session_id": session_id},
    )


async def hybrid_search(
    *,
    user_id: str | None,
    query: str,
    k_per_stream: int = _K_PER_STREAM,
    session_id: str | None = None,
) -> HybridResult:
    """Run hybrid streams in parallel, return RRF-fused candidates +
    per-stream top-1 cosine scores (used for the abstention gate).

    Scope rules:
      - user_id only: all four streams (memory + episodic) over user
      - user_id + session_id: same, but episodic restricted to session
      - session_id only: episodic-only (memory facts are user-scoped)
      - neither: empty
    """
    if not query.strip() or (not user_id and not session_id):
        return HybridResult(candidates=[])

    q_emb = await embed_one(query)
    if q_emb is None:
        # No embedding - sparse-only fallback (no abstention signal).
        sparse_mem = (
            await _memory_sparse(user_id, query, k_per_stream)
            if user_id else []
        )
        sparse_ep = await _episodic_sparse(
            user_id, query, k_per_stream, session_id=session_id,
        )
        return HybridResult(
            candidates=_fuse([], sparse_mem, [], sparse_ep),
        )

    q_vec = Vector(q_emb)
    if user_id:
        dense_mem, sparse_mem, dense_ep, sparse_ep = await asyncio.gather(
            _memory_dense(user_id, q_vec, k_per_stream),
            _memory_sparse(user_id, query, k_per_stream),
            _episodic_dense(user_id, q_vec, k_per_stream, session_id=session_id),
            _episodic_sparse(user_id, query, k_per_stream, session_id=session_id),
        )
    else:
        # session_id only: skip user-scoped memory streams
        dense_mem, sparse_mem = [], []
        dense_ep, sparse_ep = await asyncio.gather(
            _episodic_dense(None, q_vec, k_per_stream, session_id=session_id),
            _episodic_sparse(None, query, k_per_stream, session_id=session_id),
        )
    mem_top1 = float(dense_mem[0][-1]) if dense_mem else 0.0
    ep_top1 = float(dense_ep[0][-1]) if dense_ep else 0.0
    return HybridResult(
        candidates=_fuse(dense_mem, sparse_mem, dense_ep, sparse_ep),
        memory_dense_top1=mem_top1,
        episodic_dense_top1=ep_top1,
    )


def _fuse(
    dense_mem: list[tuple],
    sparse_mem: list[tuple],
    dense_ep: list[tuple],
    sparse_ep: list[tuple],
) -> list[Candidate]:
    cands: dict[str, Candidate] = {}
    fused_score: dict[str, float] = defaultdict(float)

    def add(rows: list[tuple], factory):
        for rank, row in enumerate(rows):
            c = factory(row)
            if c.key not in cands:
                cands[c.key] = c
            fused_score[c.key] += 1.0 / (_RRF_K + rank + 1)

    add(dense_mem, _candidate_from_memory)
    add(sparse_mem, _candidate_from_memory)
    add(dense_ep, _candidate_from_episodic)
    add(sparse_ep, _candidate_from_episodic)

    for key, score in fused_score.items():
        cands[key].fusion_score = score

    return sorted(cands.values(), key=lambda c: -c.fusion_score)
