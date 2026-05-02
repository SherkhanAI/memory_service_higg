"""Generate the LongMemEval comparison chart shown in the README.

Reads the per-version scores hard-coded below (kept in sync with
CHANGELOG entries) and writes ``docs/longmemeval_results.png``.

Run:
    python scripts/plot_longmemeval.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CATS = ["knowledge_update", "multi_session", "single_session", "temporal", "overall"]

# Per-CHANGELOG numbers. Add a new dict per evaluated version.
RESULTS: list[dict] = [
    {
        "label": "v0.4.5  N=12 (3/cat) seed=42",
        "color": "#94a3b8",
        "scores": {
            "knowledge_update": 1.00,
            "multi_session":    0.67,
            "single_session":   0.67,
            "temporal":         0.67,
            "overall":          0.75,
        },
    },
    {
        "label": "v0.5  N=40 (10/cat) seed=42 [402-degraded]",
        "color": "#cbd5e1",
        "scores": {
            "knowledge_update": 0.80,
            "multi_session":    0.55,
            "single_session":   0.50,
            "temporal":         0.56,
            "overall":          0.61,
        },
    },
    {
        "label": "v0.5+  N=12 (3/cat) seed=42  [+ROI: gate, prefix, source repair]  <-- submitted",
        "color": "#2563eb",
        "scores": {
            "knowledge_update": 0.83,
            "multi_session":    0.33,
            "single_session":   0.67,
            "temporal":         1.00,
            "overall":          0.71,
        },
    },
    {
        "label": "v0.6  N=12 (3/cat) seed=42  [dual-track: prefix in BM25 only]  REJECTED",
        "color": "#f87171",
        "scores": {
            "knowledge_update": 0.83,
            "multi_session":    0.33,
            "single_session":   0.33,
            "temporal":         0.83,
            "overall":          0.58,
        },
    },
]


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "docs" / "longmemeval_results.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    n_cats = len(CATS)
    n_runs = len(RESULTS)
    width = 0.78 / n_runs
    x = np.arange(n_cats)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, run in enumerate(RESULTS):
        ys = [run["scores"][c] for c in CATS]
        offset = (i - (n_runs - 1) / 2) * width
        bars = ax.bar(
            x + offset, ys, width,
            label=run["label"],
            color=run["color"],
            edgecolor="#1f2937",
            linewidth=0.8,
        )
        for b, y in zip(bars, ys):
            ax.text(
                b.get_x() + b.get_width() / 2, y + 0.015,
                f"{y:.2f}",
                ha="center", va="bottom",
                fontsize=9, color="#111827",
            )

    ax.axhline(0.5, color="#fca5a5", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0.75, color="#86efac", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(
        n_cats - 0.45, 0.755, "v0.4.5 baseline",
        fontsize=8, color="#15803d", ha="right",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CATS, rotation=12, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("LLM-as-judge score (1.0 = yes, 0.5 = partial, 0 = no)")
    ax.set_title(
        "LongMemEval-S cleaned - per-category scores by version\n"
        "(higher is better; LLM-judge on yes/partial/no verdict)",
        fontsize=11, pad=14,
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
