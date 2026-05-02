from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ..auth import verify_token
from ..schemas import SearchHit, SearchIn, SearchOut
from ..services.reranker import rerank
from ..services.retrieval import hybrid_search

router = APIRouter()
log = logging.getLogger(__name__)


@router.post(
    "/search",
    response_model=SearchOut,
    dependencies=[Depends(verify_token)],
)
async def post_search(req: SearchIn) -> SearchOut:
    """v0.4 search: hybrid retrieval + rerank, structured response.

    Same pipeline as /recall up to the reranker; skips the tiered
    assembly and instead returns top-N candidates as ``SearchHit``s.
    """
    if not req.user_id and not req.session_id:
        # Search is user- or session-scoped to avoid cross-user leaks.
        return SearchOut(results=[])

    user_id = req.user_id or ""
    if not user_id:
        # session-only scope is unusual but supported via direct SQL —
        # for v0.4 we keep it simple and require user_id for ranking.
        return SearchOut(results=[])

    hybrid = await hybrid_search(
        user_id=user_id,
        query=req.query,
        k_per_stream=req.limit * 3,
    )

    reranked = await rerank(
        query=req.query,
        candidates=hybrid.candidates,
        top_k=req.limit,
    )

    results: list[SearchHit] = []
    for c in reranked:
        score = float(c.rerank_score if c.rerank_score is not None else c.fusion_score)
        if c.source == "memory":
            sess = c.fields.get("source_turn_id") or ""
            ts = c.fields.get("t_valid")
            content = c.text
            metadata = {
                "kind": "memory",
                "predicate": c.fields.get("predicate"),
                "object_text": c.fields.get("object_text"),
            }
        else:
            sess = c.fields.get("session_id") or ""
            ts = c.fields.get("ts")
            content = c.fields.get("raw_text") or c.text
            metadata = {"kind": "episodic"}
        if not ts:
            continue
        if req.session_id and c.source == "episodic" and sess != req.session_id:
            continue
        results.append(
            SearchHit(
                content=content,
                score=score,
                session_id=sess or "",
                timestamp=ts,
                metadata=metadata,
            )
        )

    return SearchOut(results=results)
