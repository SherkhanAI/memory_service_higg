"""Real-data eval against LongMemEval-S cleaned (Wu et al., ICLR 2025).

Run:
    pytest -m longmemeval -s -k "test_longmemeval_subsample"

Configurable via env:
    LONGMEMEVAL_N_PER_CAT       default 5  — questions per category
    LONGMEMEVAL_DISTRACTORS     default 8  — random distractor sessions per Q
    LONGMEMEVAL_SEED            default 42 — for reproducibility

Skipped automatically without an OpenRouter key. Uses the same key for
both turn extraction (during ingestion) and the LLM-judge.

Cost: ~$0.10-0.20 per question (depends on session length and
extraction calls). At default n=5 per category × 5 = 25 questions, the
run is roughly $3-5 and ~60-90 minutes wall clock against
the live containerised service.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import httpx
import pytest

import re

from src.eval import judge as judge_mod
from src.eval import loader

# LongMemEval dates: "2023/05/20 (Sat) 02:21" → "2023-05-20T02:21:00Z"
_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})\s+\([^)]+\)\s+(\d{2}):(\d{2})")


def _to_iso(date_str: str) -> str:
    if "T" in date_str:
        return date_str
    m = _DATE_RE.match(date_str.strip())
    if m:
        y, mo, d, h, mi = m.groups()
        return f"{y}-{mo}-{d}T{h}:{mi}:00Z"
    # Fallback: tolerate plain YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str.strip()):
        return f"{date_str.strip()}T12:00:00Z"
    return "2024-01-01T00:00:00Z"

log = logging.getLogger(__name__)

pytestmark = pytest.mark.longmemeval


_DEFAULT_N = int(os.environ.get("LONGMEMEVAL_N_PER_CAT", "5"))
_DEFAULT_DISTRACTORS = int(os.environ.get("LONGMEMEVAL_DISTRACTORS", "8"))
_DEFAULT_SEED = int(os.environ.get("LONGMEMEVAL_SEED", "42"))

_REPORT = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "longmemeval_baseline.json"
)


def _has_key() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


@pytest.fixture(scope="module")
def dataset() -> list[dict]:
    data = loader.load()
    return loader.subsample(
        data, n_per_category=_DEFAULT_N, seed=_DEFAULT_SEED,
    )


def _ingest_question(client: httpx.Client, q: dict) -> str:
    """Returns user_id (per-question, isolated)."""
    user_id = f"lme_{q['question_id'][:24]}".replace("-", "_")
    # Cleanup any prior state
    client.delete(f"/users/{user_id}")
    session_ids, dates, sessions = loader.truncate_history(
        q, distractor_quota=_DEFAULT_DISTRACTORS,
    )
    for sid, date, msgs in zip(session_ids, dates, sessions):
        # ``msgs`` is list[{role, content}]. Send as one /turns per session.
        # If the session is huge, split into halves to respect 60s budget.
        chunks = [msgs[i:i + 8] for i in range(0, len(msgs), 8)]
        ts = _to_iso(date)
        for chunk in chunks:
            r = client.post(
                "/turns",
                json={
                    "session_id": str(sid),
                    "user_id": user_id,
                    "messages": chunk,
                    "timestamp": ts,
                    "metadata": {"longmemeval_qid": q["question_id"]},
                },
                timeout=120.0,
            )
            if r.status_code != 201:
                log.warning(
                    "ingest failed qid=%s sid=%s status=%s body=%s",
                    q["question_id"], sid, r.status_code, r.text[:200],
                )
    return user_id


def _recall_for(client: httpx.Client, q: dict, user_id: str) -> dict:
    r = client.post(
        "/recall",
        json={
            "query": q["question"],
            "session_id": f"probe_{q['question_id']}",
            "user_id": user_id,
            "max_tokens": 1024,
        },
        timeout=60.0,
    )
    r.raise_for_status()
    return r.json()


@pytest.mark.skipif(
    not _has_key(),
    reason="No OpenRouter key — skipping LongMemEval real eval",
)
def test_longmemeval_subsample(client: httpx.Client, dataset: list[dict]):
    print(f"\n=== LongMemEval-S real eval | n={len(dataset)} questions ===")

    scored: list[dict] = []
    by_cat: dict[str, list[dict]] = defaultdict(list)
    t0 = time.monotonic()

    try:
        for i, q in enumerate(dataset, 1):
            cat = loader.categorize(q.get("question_type", ""))
            qid = q["question_id"]
            print(
                f"\n[{i}/{len(dataset)}] {cat} {qid} "
                f"(answer_sessions={len(q.get('answer_session_ids') or [])} "
                f"haystack={len(q.get('haystack_session_ids') or [])})"
            )

            t_ingest = time.monotonic()
            user_id = _ingest_question(client, q)
            t_ingest = time.monotonic() - t_ingest

            t_recall = time.monotonic()
            recall = _recall_for(client, q, user_id)
            t_recall = time.monotonic() - t_recall

            verdict, score, reason = asyncio.run(
                judge_mod.judge(
                    question=q["question"],
                    gold_answer=q.get("answer", ""),
                    system_context=recall.get("context", ""),
                )
            )
            entry = {
                "qid": qid,
                "category": cat,
                "verdict": verdict,
                "score": score,
                "ingest_s": round(t_ingest, 1),
                "recall_s": round(t_recall, 1),
                "ctx_len": len(recall.get("context") or ""),
                "n_citations": len(recall.get("citations") or []),
                "reason": reason[:200],
            }
            scored.append(entry)
            by_cat[cat].append(entry)
            print(
                f"   ingest={entry['ingest_s']}s recall={entry['recall_s']}s "
                f"verdict={verdict} score={score} ctx_len={entry['ctx_len']}"
            )
            # Cleanup user to keep DB lean for next iteration
            client.delete(f"/users/{user_id}")
    finally:
        # Best-effort cleanup
        for entry in scored:
            try:
                client.delete(f"/users/lme_{entry['qid'][:24]}".replace("-", "_"))
            except Exception:
                pass

    # Aggregate — exclude judge errors (score < 0) from the average
    # so infra failures don't bias the metric down.
    elapsed = time.monotonic() - t0
    valid = [e for e in scored if e["score"] >= 0]
    judge_errors = len(scored) - len(valid)
    overall = sum(e["score"] for e in valid) / max(1, len(valid))
    per_cat = {}
    for cat, items in by_cat.items():
        valid_items = [e for e in items if e["score"] >= 0]
        per_cat[cat] = (
            sum(e["score"] for e in valid_items) / len(valid_items)
            if valid_items else 0.0
        )

    print("\n=== LongMemEval results ===")
    for cat in sorted(by_cat):
        valid_items = [e for e in by_cat[cat] if e["score"] >= 0]
        print(f"  {cat:18s}  {per_cat[cat]:.2f}  ({len(valid_items)}/{len(by_cat[cat])} q valid)")
    print(f"  {'overall':18s}  {overall:.2f}  ({len(valid)}/{len(scored)} q valid)")
    if judge_errors:
        print(f"  {'judge_errors':18s}  {judge_errors}  (excluded from average)")
    print(f"  wall_clock          {elapsed/60:.1f} min")

    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    _REPORT.write_text(
        json.dumps(
            {
                "n_per_category": _DEFAULT_N,
                "distractors": _DEFAULT_DISTRACTORS,
                "seed": _DEFAULT_SEED,
                "overall": overall,
                "per_category": per_cat,
                "elapsed_s": round(elapsed, 1),
                "details": scored,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nbaseline written: {_REPORT}")
