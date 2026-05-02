from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ..auth import verify_token
from ..schemas import RecallIn, RecallOut
from ..services.assembler import (
    assemble,
    fetch_recent_turns,
    fetch_stable_facts,
)
from ..services.reranker import rerank
from ..services.retrieval import hybrid_search

router = APIRouter()
log = logging.getLogger(__name__)


_TOP_K_PER_STREAM = 30
_RERANK_TOP_K = 10


@router.post(
    "/recall",
    response_model=RecallOut,
    dependencies=[Depends(verify_token)],
)
async def post_recall(req: RecallIn) -> RecallOut:
    """v0.4 recall pipeline.

    1. Hybrid retrieve memory+episodic (4 streams, RRF fused).
    2. Cohere rerank top-30 → top-10.
    3. Fetch stable identity facts (predicate-filtered, no retrieval).
    4. Fetch last few turns of current session.
    5. Tiered assembly under ``max_tokens``: stable → relevant → recent.
    6. Abstention gate: empty body if reranker top-1 below threshold.
    """
    if not req.user_id:
        return RecallOut(context="", citations=[])

    hybrid = await hybrid_search(
        user_id=req.user_id,
        query=req.query,
        k_per_stream=_TOP_K_PER_STREAM,
    )

    reranked = await rerank(
        query=req.query,
        candidates=hybrid.candidates,
        top_k=_RERANK_TOP_K,
    )

    stable = await fetch_stable_facts(req.user_id)
    recent = await fetch_recent_turns(req.session_id, limit=4)

    body, citations = assemble(
        stable_facts=stable,
        relevant=reranked,
        recent=recent,
        max_tokens=req.max_tokens,
        memory_dense_top1=hybrid.memory_dense_top1,
        episodic_dense_top1=hybrid.episodic_dense_top1,
    )
    return RecallOut(context=body, citations=citations)
