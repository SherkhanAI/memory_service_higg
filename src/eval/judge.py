"""LLM-as-judge for LongMemEval recall quality.

Calls OpenRouter chat completion (default ``settings.extraction_model``,
which is ``openai/gpt-5.4-mini``) with a strict JSON schema asking
whether the system's recall context is sufficient to answer the gold
question.

For abstention questions the gold answer typically reads "the user has
not mentioned this" or similar, and we additionally require the system
context to be empty — answering with stable facts on an unrelated
question is a fail.
"""

from __future__ import annotations

import logging
from typing import Literal

from ..services.llm import chat_json

log = logging.getLogger(__name__)


_VERDICT_SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0}

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"enum": ["yes", "partial", "no"]},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "reasoning"],
}


def _judge_prompt() -> str:
    return """You are grading a memory-recall system for a personal AI assistant.

You will see:
  - QUESTION  — what the user asked the assistant
  - GOLD      — the verified correct answer
  - CONTEXT   — what the recall system surfaced for the assistant to use

Decide whether CONTEXT contains enough information for the assistant to
correctly answer the QUESTION:

  - "yes"     : CONTEXT clearly conveys the gold answer (verbatim or paraphrase).
  - "partial" : CONTEXT contains related info but is missing the exact answer
                (assistant could give a degraded but still on-topic response).
  - "no"      : CONTEXT is empty, irrelevant, or contradicts the gold answer.

Special rule for abstention questions (where the gold answer is something
like "no, the user has not mentioned this", "I don't know", "the user
hasn't shared that"):
  - "yes" iff CONTEXT is empty or the listed facts genuinely do not
    cover the topic of the question.
  - "no"  if CONTEXT presents unrelated user facts as if they were
    relevant (assistant would hallucinate).

Output JSON exactly: {"verdict": "yes"|"partial"|"no", "reasoning": "..."}"""


async def judge(question: str, gold_answer: str, system_context: str) -> tuple[str, float, str]:
    """Returns (verdict, score, reasoning).

    On judge failure returns ``("error", -1.0, ...)``; the runner
    excludes these from the average rather than counting them as
    failures (which would bias the metric down).
    """
    # Truncate excessive context — some haystacks are huge and we
    # don't want the judge to run out of context budget.
    ctx = (system_context or "(empty)")[:6000]
    user = (
        f"QUESTION: {question}\n\n"
        f"GOLD: {gold_answer}\n\n"
        f"CONTEXT:\n---\n{ctx}\n---\n\n"
        f"Grade the recall context."
    )

    last_err = "no_result"
    for attempt in range(3):
        result = await chat_json(
            system=_judge_prompt(),
            user=user,
            schema=_SCHEMA,
            schema_name="judge",
            temperature=0.0,
            timeout_s=45.0,
        )
        if result and "verdict" in result:
            verdict = result["verdict"]
            reasoning = result.get("reasoning", "")
            return verdict, _VERDICT_SCORE.get(verdict, 0.0), reasoning
        last_err = f"empty_result_attempt_{attempt}"
    return "error", -1.0, last_err
