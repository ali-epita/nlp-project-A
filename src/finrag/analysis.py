"""Loading and summarising stored runs for the analysis scripts and notebook."""

from __future__ import annotations

import numpy as np

from .config import GENERATIONS_DIR, RUNS_DIR
from .io import read_json


def load_retrieval_runs() -> list[dict]:
    return sorted(
        (read_json(p) for p in RUNS_DIR.glob("*.json")),
        key=lambda r: r["summary"]["recall"],
        reverse=True,
    )


def load_generation_runs() -> list[dict]:
    return [read_json(p) for p in sorted(GENERATIONS_DIR.glob("*.json"))]


def _rate(values: list) -> float | None:
    known = [v for v in values if v is not None]
    return float(np.mean(known)) if known else None


def summarize_generation(record: dict, judge_name: str | None = None) -> dict:
    """Aggregate deterministic scores and, when present, judge verdicts."""
    gens = record["generations"]
    scores = [g["scores"] for g in gens]
    summary = {
        "n_questions": len(gens),
        "abstention_rate": _rate([s["abstained"] for s in scores]),
        "citation_rate": _rate([s["has_citation"] for s in scores]),
        "citation_valid_rate": _rate([s["citation_valid"] for s in scores]),
        "numeric_accuracy": _rate([s["numeric_correct"] for s in scores]),
        "numeric_accuracy_scale_tolerant": _rate(
            [s["numeric_correct_scale_tolerant"] for s in scores]
        ),
        "token_f1": _rate([s["token_f1"] for s in scores]),
        "mean_seconds_per_question": _rate([g["seconds"] for g in gens]),
        "n_truncation_suspected": sum(bool(g.get("truncation_suspected")) for g in gens),
    }

    names = set()
    for g in gens:
        names.update(g.get("verdicts", {}))
    for name in sorted(names):
        if judge_name is not None and name != judge_name:
            continue
        verdicts = [g["verdicts"][name] for g in gens if name in g.get("verdicts", {})]
        correctness = [v["correctness"] for v in verdicts]
        groundedness = [v["groundedness"] for v in verdicts if v["groundedness"] is not None]
        summary[f"judge:{name}"] = {
            "n_judged": len(verdicts),
            "mean_correctness": _rate(correctness),
            "fully_correct_rate": _rate([c == 2 for c in correctness]),
            "partially_correct_rate": _rate([c == 1 for c in correctness]),
            "mean_groundedness": _rate(groundedness),
            "fully_grounded_rate": _rate([g == 2 for g in groundedness]),
            "hallucination_rate": _rate([v["hallucination"] for v in verdicts]),
            "citations_supported_rate": _rate([v["citations_supported"] for v in verdicts]),
        }
    return summary


def primary_judge_name(record: dict) -> str | None:
    """The judge present on the most generations of a run (ollama preferred)."""
    counts: dict[str, int] = {}
    for g in record["generations"]:
        for name in g.get("verdicts", {}):
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    ollama = {n: c for n, c in counts.items() if n.startswith("ollama:")}
    pool = ollama or counts
    return max(pool, key=pool.get)
