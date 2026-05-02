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
from datetime import datetime

from ..config import settings
from . import predicates
from .llm import chat_json

log = logging.getLogger(__name__)


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
    for f in facts:
        if not isinstance(f, dict):
            continue
        if not f.get("predicate") or not f.get("object_text"):
            continue
        if f.get("stance") == "none":
            f["stance"] = None
        cleaned.append(f)
    log.info("extracted %d/%d facts", len(cleaned), len(facts))
    return cleaned
