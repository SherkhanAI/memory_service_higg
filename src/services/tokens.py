"""Token counting and budget-aware joining via tiktoken.

We count with the cl100k_base encoder (used by gpt-4 / gpt-4o / gpt-5
families) as a stable approximation across providers. Approximate is
fine — ТЗ allows ~factor-of-1 overshoot but not 2×.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count(text: str) -> int:
    if not text:
        return 0
    return len(_encoder().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    enc = _encoder()
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])


def join_within_budget(parts: list[str], max_tokens: int, sep: str = "\n") -> tuple[str, int]:
    """Greedy join of parts; stops at max_tokens. Returns (joined, used_tokens)."""
    enc = _encoder()
    sep_cost = len(enc.encode(sep))
    used = 0
    out: list[str] = []
    for p in parts:
        cost = len(enc.encode(p))
        extra = cost + (sep_cost if out else 0)
        if used + extra > max_tokens:
            break
        out.append(p)
        used += extra
    return sep.join(out), used
