"""LongMemEval-shaped self-eval fixture runner.

Runs against the live service. Skipped automatically when no
``OPENROUTER_API_KEY`` (or any embedding key) is configured, since recall
quality is not measurable without real embeddings.

Usage:
    docker compose up -d
    pytest -m memeval -s           # -s prints the per-category table

Six categories:
    recall       — single-hop fact retrieval
    multi_hop    — chain across 2+ memories
    temporal     — ordering/before/after
    supersession — current vs stale fact (Stripe → Notion)
    abstention   — topic never discussed → must return empty
    isolation    — cross-user leakage check

Score per probe (0 or 1):
    expected_empty=True   → empty context wins
    expected_keywords     → at least one keyword must appear (case-insens.)
    forbidden_keywords    → none may appear (zero out otherwise)

The aggregate is recorded in tests/fixtures/memeval_baseline.json on the
first run. Subsequent runs fail if any category drops > 5pp from baseline.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections import defaultdict
from pathlib import Path

import httpx
import pytest


pytestmark = pytest.mark.memeval


_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
_BASELINE = Path(__file__).resolve().parent / "fixtures" / "memeval_baseline.json"
_BASELINE.parent.mkdir(parents=True, exist_ok=True)
_TOLERANCE = 0.05


def _has_key() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY")) or bool(
        os.environ.get("OPENAI_API_KEY")
    ) or bool(os.environ.get("GOOGLE_API_KEY"))


@pytest.fixture(scope="module")
def fixtures():
    convs = json.loads((_FIXTURES / "conversations.json").read_text("utf-8"))
    probes = json.loads((_FIXTURES / "probes.json").read_text("utf-8"))
    return convs, probes


def _ingest(client: httpx.Client, user: dict) -> dict[str, str]:
    """Returns {session_id: last_turn_id} for probe synchronisation."""
    last_turn_per_session: dict[str, str] = {}
    user_id = user["user_id"]
    for session in user["sessions"]:
        sid = session["session_id"]
        ts = session["ts"]
        for turn in session["turns"]:
            r = client.post(
                "/turns",
                json={
                    "session_id": sid,
                    "user_id": user_id,
                    "messages": turn["messages"],
                    "timestamp": ts,
                    "metadata": {},
                },
                timeout=60.0,
            )
            assert r.status_code == 201, f"ingest failed: {r.status_code} {r.text}"
            last_turn_per_session[sid] = r.json()["id"]
    return last_turn_per_session


def _score_probe(probe: dict, body: dict) -> int:
    ctx = (body.get("context") or "").lower()
    cits = body.get("citations") or []
    if probe.get("expected_empty"):
        # No context AND no citations
        return 1 if (not ctx.strip() and not cits) else 0
    expected = [k.lower() for k in probe.get("expected_keywords", [])]
    forbidden = [k.lower() for k in probe.get("forbidden_keywords", [])]
    if any(re.search(re.escape(f), ctx) for f in forbidden):
        return 0
    if not expected:
        return 1
    return 1 if any(re.search(re.escape(e), ctx) for e in expected) else 0


def _aggregate(scored: list[tuple[dict, int]]) -> dict[str, float]:
    by_cat: dict[str, list[int]] = defaultdict(list)
    for probe, s in scored:
        by_cat[probe["category"]].append(s)
    agg = {cat: sum(v) / len(v) for cat, v in by_cat.items()}
    agg["_overall"] = sum(s for _, s in scored) / max(1, len(scored))
    return agg


def _print_report(agg: dict[str, float], scored: list[tuple[dict, int]]) -> None:
    print("\n=== memeval ===")
    for cat in sorted(agg):
        if cat == "_overall":
            continue
        n = sum(1 for p, _ in scored if p["category"] == cat)
        print(f"  {cat:14s}  {agg[cat]:.2f}  ({int(agg[cat] * n)}/{n})")
    print(f"  {'overall':14s}  {agg['_overall']:.2f}")


@pytest.mark.skipif(not _has_key(), reason="No embedding key — skipping memeval")
def test_memeval(client: httpx.Client, fixtures) -> None:
    convs, probes_doc = fixtures
    probes = probes_doc["probes"]

    # Cleanup
    for u in convs["users"]:
        client.delete(f"/users/{u['user_id']}")

    # Ingest in interleaved order: walk session-by-session for each user,
    # firing probes that target each session as we finish it.
    probes_by_session: dict[str, list[dict]] = defaultdict(list)
    for p in probes:
        probes_by_session[p["probe_after_session"]].append(p)

    scored: list[tuple[dict, int]] = []
    try:
        for user in convs["users"]:
            user_id = user["user_id"]
            for session in user["sessions"]:
                sid = session["session_id"]
                ts = session["ts"]
                for turn in session["turns"]:
                    r = client.post(
                        "/turns",
                        json={
                            "session_id": sid,
                            "user_id": user_id,
                            "messages": turn["messages"],
                            "timestamp": ts,
                            "metadata": {},
                        },
                        timeout=60.0,
                    )
                    assert r.status_code == 201, r.text
                # Run probes scheduled after this session.
                for probe in probes_by_session.get(sid, []):
                    rr = client.post(
                        "/recall",
                        json={
                            "query": probe["query"],
                            "session_id": f"probe_{uuid.uuid4().hex[:6]}",
                            "user_id": probe["user_id"],
                            "max_tokens": 512,
                        },
                        timeout=30.0,
                    )
                    assert rr.status_code == 200, rr.text
                    body = rr.json()
                    s = _score_probe(probe, body)
                    scored.append((probe, s))
                    if s == 0:
                        ctx = (body.get("context") or "")[:800].replace("\n", " | ")
                        cits = body.get("citations") or []
                        top_score = cits[0]["score"] if cits else 0.0
                        print(f"\n  FAIL [{probe['category']}] {probe['id']}: {probe['query']}")
                        print(f"        expected={probe.get('expected_keywords')} forbidden={probe.get('forbidden_keywords')} expected_empty={probe.get('expected_empty')}")
                        print(f"        top_score={top_score:.3f}  ncits={len(cits)}")
                        print(f"        ctx={ctx[:600]!r}")
    finally:
        for u in convs["users"]:
            client.delete(f"/users/{u['user_id']}")

    agg = _aggregate(scored)
    _print_report(agg, scored)

    # Baseline gate
    if _BASELINE.exists():
        baseline = json.loads(_BASELINE.read_text("utf-8"))
        regressions = []
        for cat, score in agg.items():
            if cat in baseline and score + _TOLERANCE < baseline[cat]:
                regressions.append(
                    f"{cat}: {baseline[cat]:.2f} → {score:.2f} (drop "
                    f"{baseline[cat] - score:.2f})"
                )
        assert not regressions, "memeval regressions:\n  " + "\n  ".join(regressions)
    else:
        _BASELINE.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print(f"\nbaseline written: {_BASELINE}")
