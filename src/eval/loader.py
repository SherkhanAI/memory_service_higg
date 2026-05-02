"""LongMemEval-S cleaned dataset loader.

Single 277 MB JSON file, downloaded once from
``https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned``,
cached at ``fixtures/longmemeval/longmemeval_s_cleaned.json``.

We subsample by category for tractable iteration. Each question has
~48 sessions / ~115K tokens of haystack history; full ingestion of
even 25 questions takes hours and many dollars. We truncate to
``answer_session_ids`` (evidence) + a small random sample of
distractors so the recall signal survives without paying for the
full noise floor on every iteration.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

_HF_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_s_cleaned.json"
)
_CACHE = Path(__file__).resolve().parent.parent.parent / "fixtures" / "longmemeval" / "longmemeval_s_cleaned.json"


# Map LongMemEval question_type → our category buckets used in
# CHANGELOG. Abstention is detected by ``_abs`` suffix.
_CATEGORY_MAP = {
    "single-session-user": "single_session",
    "single-session-assistant": "single_session",
    "single-session-preference": "single_session",
    "multi-session": "multi_session",
    "temporal-reasoning": "temporal",
    "knowledge-update": "knowledge_update",
}


def categorize(question_type: str) -> str:
    if question_type.endswith("_abs"):
        return "abstention"
    return _CATEGORY_MAP.get(question_type, "other")


def ensure_dataset() -> Path:
    """Download once, return cached path."""
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    if _CACHE.exists() and _CACHE.stat().st_size > 1_000_000:
        return _CACHE
    log.info("downloading LongMemEval-S cleaned (~277 MB)...")
    with httpx.stream(
        "GET", _HF_URL, follow_redirects=True, timeout=600.0
    ) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        bytes_read = 0
        last_pct = -10
        with _CACHE.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 256):
                f.write(chunk)
                bytes_read += len(chunk)
                if total:
                    pct = int(bytes_read * 100 / total)
                    if pct - last_pct >= 10:
                        log.info("  ... %d%% (%d MB / %d MB)",
                                 pct, bytes_read // 1_048_576, total // 1_048_576)
                        last_pct = pct
    log.info("cached at %s (%d MB)", _CACHE, _CACHE.stat().st_size // 1_048_576)
    return _CACHE


def load() -> list[dict]:
    path = ensure_dataset()
    return json.loads(path.read_text(encoding="utf-8"))


def subsample(
    data: list[dict],
    n_per_category: int = 5,
    seed: int = 42,
    skip_categories: tuple[str, ...] = ("other",),
) -> list[dict]:
    """Stratified subsample with deterministic seed."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for q in data:
        cat = categorize(q.get("question_type", ""))
        if cat in skip_categories:
            continue
        by_cat[cat].append(q)
    rng = random.Random(seed)
    out: list[dict] = []
    for cat in sorted(by_cat):
        items = by_cat[cat][:]
        rng.shuffle(items)
        out.extend(items[:n_per_category])
    return out


def truncate_history(
    question: dict,
    distractor_quota: int = 8,
    seed: int | None = None,
) -> tuple[list[str], list[str], list[list[dict]]]:
    """Returns (session_ids, dates, sessions) — all chronological,
    containing every evidence session plus a small random sample of
    distractors capped at ``distractor_quota``.
    """
    session_ids = question.get("haystack_session_ids") or []
    dates = question.get("haystack_dates") or []
    sessions = question.get("haystack_sessions") or []
    answer_ids = set(question.get("answer_session_ids") or [])

    rng = random.Random(seed if seed is not None else question.get("question_id", "x"))
    keep_idx: set[int] = set()
    distractors: list[int] = []
    for i, sid in enumerate(session_ids):
        if sid in answer_ids:
            keep_idx.add(i)
        else:
            distractors.append(i)
    rng.shuffle(distractors)
    for i in distractors[:distractor_quota]:
        keep_idx.add(i)

    out_ids = [session_ids[i] for i in sorted(keep_idx)]
    out_dates = [dates[i] for i in sorted(keep_idx)] if dates else []
    out_sessions = [sessions[i] for i in sorted(keep_idx)]
    return out_ids, out_dates, out_sessions
