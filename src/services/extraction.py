"""Single-call structured extractor.

Given a turn's flattened text, ask the LLM to emit a list of facts
matching ``FACT_SCHEMA`` (strict JSON Schema). The output is consumed
by ``reconciliation.reconcile_and_write``.

We deliberately keep ``object_qualifiers`` out of the strict schema —
strict mode requires every property to be enumerated, and qualifiers
are open-ended. Anything subordinate (breed, role qualifier, dates)
goes into ``source_text`` and is recovered by the reconciliation prompt
when needed.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from difflib import SequenceMatcher

from ..config import settings
from . import predicates
from .llm import chat_json

log = logging.getLogger(__name__)


def _normalise(s: str) -> str:
    """Lower-case, collapse whitespace; for substring comparison only."""
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _repair_source_text(source_text: str, turn_text: str, object_text: str) -> str:
    """Anchor source_text to the actual turn text.

    Why: the LLM frequently paraphrases the supporting span ("I work at
    Notion as a PM" -> "user works at Notion"), which strips the exact
    dates / numbers / phrasings that the LongMemEval judge keys on.

    Strategy (cheapest-first):
      1. If source_text appears verbatim (case-insensitive, whitespace
         normalised) in turn_text, keep the original phrasing from the
         turn (preserves casing).
      2. If object_text appears in turn_text, return a 200-char window
         centred on it - this preserves the exact phrasing around the
         answer fact, which is what the judge needs.
      3. Else find the longest common substring between source_text and
         turn_text (>=15 chars). If found, return the matching span
         from turn_text.
      4. Fallback: return the original source_text (paraphrased).
    """
    if not source_text or not turn_text:
        return source_text or ""
    norm_source = _normalise(source_text)
    norm_turn = _normalise(turn_text)
    # 1. Verbatim hit (after normalisation): recover original casing.
    pos = norm_turn.find(norm_source)
    if pos >= 0:
        # Map normalised position back to turn_text approximately. Cheap
        # heuristic: walk turn_text by char ignoring whitespace.
        return _slice_original(turn_text, norm_turn, pos, len(norm_source))
    # 2. Object_text anchor.
    norm_object = _normalise(object_text)
    if norm_object and len(norm_object) >= 3:
        opos = norm_turn.find(norm_object)
        if opos >= 0:
            start = max(0, opos - 80)
            end = min(len(norm_turn), opos + len(norm_object) + 120)
            window = _slice_original(turn_text, norm_turn, start, end - start)
            if window:
                return window[:200]
    # 3. Longest common substring.
    matcher = SequenceMatcher(None, norm_source, norm_turn, autojunk=False)
    m = matcher.find_longest_match(0, len(norm_source), 0, len(norm_turn))
    if m.size >= 15:
        return _slice_original(turn_text, norm_turn, m.b, m.size)
    # 4. Give up - keep the paraphrase rather than drop the fact.
    return source_text[:200]


def _slice_original(turn_text: str, norm_turn: str, norm_start: int, norm_len: int) -> str:
    """Approximate inverse of _normalise: walk turn_text, count
    significant chars, return the original-casing slice that contains
    norm_len normalised chars starting at norm_start."""
    if norm_start < 0 or norm_len <= 0:
        return ""
    consumed = 0
    orig_start: int | None = None
    orig_end: int | None = None
    prev_ws = False
    norm_pos = 0
    for i, ch in enumerate(turn_text):
        is_ws = ch.isspace()
        if is_ws:
            if prev_ws:
                continue
            ch_norm = " "
        else:
            ch_norm = ch.lower()
        prev_ws = is_ws
        if norm_pos == norm_start and orig_start is None:
            orig_start = i
        if orig_start is not None and consumed >= norm_len:
            orig_end = i
            break
        if orig_start is not None:
            consumed += 1
        norm_pos += 1
    if orig_start is None:
        return ""
    if orig_end is None:
        orig_end = len(turn_text)
    return turn_text[orig_start:orig_end].strip()[:200]


_FACT_ITEM = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "predicate": {"type": "string"},
        "object_text": {"type": "string"},
        "kind": {"enum": ["fact", "preference", "opinion", "event"]},
        "stance": {"enum": ["positive", "negative", "neutral", "none"]},
        "confidence": {"enum": ["low", "med", "high"]},
        "is_implicit": {"type": "boolean"},
        "source_text": {"type": "string"},
    },
    "required": [
        "predicate", "object_text", "kind", "stance",
        "confidence", "is_implicit", "source_text",
    ],
}

EXTRACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "facts": {"type": "array", "items": _FACT_ITEM},
    },
    "required": ["facts"],
}


def _system_prompt() -> str:
    return f"""You are a memory curator for a long-term agent memory system. \
Given the latest conversation turn (and optional context from the same \
session), extract durable facts about THE USER (not the agent or \
unrelated third parties).

For each fact emit:
- predicate: pick from the canonical list below, OR use "other:<short_descriptor>" if nothing fits.
- object_text: the concrete value — an entity NAME or attribute value (e.g. "Notion", "Berlin", "Biscuit", "TypeScript", "golden retriever", "vegetarian"). NEVER a pronoun. NEVER a phrase ("he", "the dog", "my company").
- kind: fact | preference | opinion | event.
- stance: positive | negative | neutral (only meaningful for kind=opinion; use "none" otherwise).
- confidence: high (directly stated, unambiguous) | med (clearly implied) | low (vague hint).
- is_implicit: true if inferred rather than stated outright.
- source_text: the EXACT phrase (≤200 chars) from the turn supporting this fact.

Hard rules:
1. **No pronouns or vague references in object_text**. If the turn says "He's a golden retriever" and "he" wasn't named in this turn, either resolve via SESSION CONTEXT below or skip the fact entirely. NEVER emit object_text="he", "she", "it", "they", "the dog", "my company".
2. For attributive descriptions of an already-known entity (e.g. previous turn named the pet), emit a fact about the **attribute** (breed, age) under a sensible predicate — not a duplicate identity fact. For "He's a golden retriever" with pet=Biscuit known: emit nothing (breed is an attribute, not a durable predicate in our list) OR skip.
3. Skip ephemeral states ("I'm tired today", "I'm at the airport now"), small talk, and facts about the agent.
4. For life changes (job change, move, breakup): emit only the NEW state. Do NOT also emit the old state unless the user explicitly tags it as past ("I used to work at Stripe").
5. For corrections ("actually I meant X, not Y"): emit only the corrected version.
6. For opinions (kind=opinion): object_text = the topic name (e.g. "TypeScript"); set stance accordingly. The CURRENT belief is what matters.
7. Implicit-fact examples — DO extract these:
   - "walking Biscuit this morning" → predicate=pet, object_text=Biscuit, is_implicit=true, confidence=med
   - "from my apartment in SF" → predicate=lives_in, object_text=San Francisco, is_implicit=true, confidence=med
8. Multi-fact turns: emit one entry per fact. A single sentence can yield multiple facts.
9. If nothing extractable, return {{"facts": []}}. Do not invent.

CANONICAL PREDICATES:
{predicates.predicate_glossary()}

Respond with JSON matching the schema EXACTLY. No prose, no markdown."""


async def extract_facts(
    turn_text: str,
    turn_ts: datetime,
    *,
    session_context: str | None = None,
) -> list[dict]:
    """Returns a list of fact dicts, possibly empty.

    ``session_context`` is the prior turn(s) of the same session, used
    to resolve coreferences ("he", "the dog") to named entities.
    """
    if not settings.has_extraction_key:
        return []
    if not (turn_text or "").strip():
        return []

    ctx_block = (
        f"PRIOR TURNS IN THIS SESSION (for coreference only — do not extract from):\n"
        f"---\n{session_context}\n---\n\n"
        if session_context else ""
    )
    user = (
        f"Turn timestamp: {turn_ts.isoformat()}\n"
        f"{ctx_block}"
        f"CURRENT TURN (extract from this):\n---\n{turn_text}\n---\n\nExtract facts."
    )
    result = await chat_json(
        system=_system_prompt(),
        user=user,
        schema=EXTRACT_SCHEMA,
        schema_name="extracted_facts",
    )
    if not result:
        return []
    facts = result.get("facts") or []
    if not isinstance(facts, list):
        return []

    cleaned: list[dict] = []
    repaired = 0
    for f in facts:
        if not isinstance(f, dict):
            continue
        if not f.get("predicate") or not f.get("object_text"):
            continue
        if f.get("stance") == "none":
            f["stance"] = None
        original = f.get("source_text") or ""
        f["source_text"] = _repair_source_text(
            original, turn_text, f.get("object_text") or "",
        )
        if _normalise(original) and _normalise(f["source_text"]) != _normalise(original):
            repaired += 1
        cleaned.append(f)
    log.info(
        "extracted %d/%d facts (source_text repaired=%d)",
        len(cleaned), len(facts), repaired,
    )
    return cleaned


# --- Anthropic-style contextual prefix --------------------------------------

_PREFIX_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"prefix": {"type": "string"}},
    "required": ["prefix"],
}

_PREFIX_SYSTEM = """You write a short situating prefix for a single \
conversation turn so it embeds and indexes more precisely.

The prefix MUST:
- Be 25-60 words, one paragraph, no markdown.
- Mention the broad TOPIC of the turn (job, location, family, hobby, \
travel, health, opinion about X, etc.).
- Mention any concrete named ENTITIES from this turn (people, places, \
companies, products, dates) so they appear in the embedding text.
- If the prior session context names an entity that the current turn \
refers to with a pronoun, INCLUDE the resolved entity name in the prefix.
- NOT add information that is not in the turn or the prior context.
- NOT use first or second person ("I", "you"). Write about "the user".

Return JSON: {"prefix": "<25-60 words>"}.
"""


async def build_contextual_prefix(
    turn_text: str,
    *,
    session_context: str | None = None,
    session_id: str | None = None,
    turn_ts: datetime | None = None,
) -> str | None:
    """Generate a short situating prefix for ``turn_text``.

    Following Anthropic's Contextual Retrieval recipe (2024): each
    chunk gets a 25-60 word prefix that names the topic and entities,
    prepended BEFORE embedding and BM25 indexing. Reported -49 percent
    retrieval failure on the original eval; we apply it only on the
    episodic_turn write path.

    Returns ``None`` if the upstream LLM call fails - the caller should
    embed/index ``turn_text`` alone in that case.
    """
    if not settings.has_extraction_key:
        return None
    if not (turn_text or "").strip():
        return None
    when = turn_ts.date().isoformat() if turn_ts else "unknown"
    sess_block = (
        f"PRIOR TURNS IN THIS SESSION (for entity resolution only):\n"
        f"---\n{session_context}\n---\n\n"
        if session_context else ""
    )
    user = (
        f"Session: {session_id or '?'}  Date: {when}\n\n"
        f"{sess_block}"
        f"CURRENT TURN:\n---\n{turn_text}\n---\n\n"
        f"Write the situating prefix."
    )
    result = await chat_json(
        system=_PREFIX_SYSTEM,
        user=user,
        schema=_PREFIX_SCHEMA,
        schema_name="contextual_prefix",
        temperature=0.0,
        timeout_s=15.0,
    )
    if not result:
        return None
    prefix = (result.get("prefix") or "").strip()
    if not prefix:
        return None
    # Hard cap so it never dominates the embedding budget.
    if len(prefix) > 500:
        prefix = prefix[:500]
    return prefix
