"""Canonical predicate vocabulary for structured memories.

The extraction LLM must pick a predicate from this list (or use
``other:<short>`` if nothing fits). Normalised predicates make
contradiction detection trivial: same ``(user_id, predicate)`` with a
different ``object_text`` is a candidate for SUPERSEDE; with overlapping
qualifiers it's a candidate for UPDATE.

Three classes:
- **EXCLUSIVE** — at most one active row per ``(user, predicate)``.
  Conflict ⇒ SUPERSEDE.
- **MULTI_VALUED** — many active rows OK (e.g. multiple skills).
  Same object ⇒ NOOP, different ⇒ ADD.
- **OPINION** — special-case opinion arc tracking; SUPERSEDE on stance
  shift, NOOP on rephrase.
"""

from __future__ import annotations


CANONICAL_PREDICATES: dict[str, str] = {
    # Identity
    "name": "User's name (full or preferred).",
    "age": "User's age in years.",
    "gender": "User's stated gender or pronouns.",
    "lives_in": "City/region where the user currently lives.",
    "lived_in": "Past place of residence.",
    "born_in": "City/country of birth.",
    "nationality": "User's nationality.",
    # Work
    "employer": "Current employer / company.",
    "role": "Current job title or role.",
    "employer_past": "Past employer (only when the user explicitly mentions it as past).",
    "employer_start": "Start date / month at the current employer.",
    # Education
    "degree": "Academic degree (e.g. BSc Computer Science).",
    "school": "School / university attended.",
    "field_of_study": "Major / field.",
    # Relationships
    "spouse": "Spouse / married partner's name or detail.",
    "partner": "Romantic partner (unmarried).",
    "family.parent": "Parent — name, relation detail.",
    "family.child": "Child — name, age, relation detail.",
    "family.sibling": "Sibling — name, relation detail.",
    "friend": "Named friend.",
    # Pets
    "pet": "Pet — set object_text to the pet's name; species in source_text if mentioned.",
    # Preferences
    "preference.food": "Food preference, dietary restriction, or allergy.",
    "preference.language": "Programming language preference.",
    "preference.tool": "Tool / software preference.",
    "preference.framework": "Framework or library preference.",
    "preference.activity": "Hobby or recreational activity preference.",
    # Opinions
    "opinion_about": (
        "User's stance on a specific topic. object_text = the topic "
        "(e.g. 'TypeScript'). Use stance ∈ {positive, negative, neutral}."
    ),
    # Events
    "event.travel": "User travelled to a place.",
    "event.health": "Health-related life event.",
    "event.life_change": "Major life event (move, job change, marriage, etc).",
    # Skills
    "skill": "Professional skill.",
    "hobby": "Hobby or recreational activity.",
}


EXCLUSIVE_PREDICATES: set[str] = {
    "name", "age", "gender", "lives_in", "born_in", "nationality",
    "employer", "role", "employer_start",
    "spouse", "partner",
}

MULTI_VALUED_PREDICATES: set[str] = {
    "lived_in", "employer_past",
    "family.parent", "family.child", "family.sibling", "friend",
    "preference.food", "preference.language", "preference.tool",
    "preference.framework", "preference.activity",
    "skill", "hobby",
    "pet",
    "event.travel", "event.health", "event.life_change",
    "degree", "school", "field_of_study",
}

OPINION_PREDICATES: set[str] = {"opinion_about"}

# Predicates that count as "stable identity" — always relevant to recall context
# regardless of the specific query (used by tiered assembler in v0.4+).
STABLE_PREDICATES: set[str] = {
    "name", "age", "gender", "lives_in", "born_in", "nationality",
    "employer", "role",
    "spouse", "partner",
    "pet",
    "family.parent", "family.child", "family.sibling",
    "preference.food", "preference.language",
    "skill", "hobby",
}


def is_exclusive(predicate: str) -> bool:
    return predicate in EXCLUSIVE_PREDICATES


def is_multi_valued(predicate: str) -> bool:
    return predicate in MULTI_VALUED_PREDICATES


def is_opinion(predicate: str) -> bool:
    return predicate in OPINION_PREDICATES


def predicate_glossary() -> str:
    return "\n".join(f"- `{k}` — {v}" for k, v in CANONICAL_PREDICATES.items())
