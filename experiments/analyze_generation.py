"""Analysis of the generation runs: tables, figures, and the
retrieval-versus-generation decomposition.

The decomposition compares one model's answers under retrieved context against
the same model under oracle context. Questions the model solves with gold
evidence but misses with retrieved passages are retrieval-limited (headroom a
better retriever could recover); questions missed even with gold evidence are
generation-limited (the model's own ceiling).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from finrag.analysis import load_generation_runs, primary_judge_name, summarize_generation
from finrag.config import FIGURES, TABLES
from finrag.plotting import INK, PALETTE, bar_labels, save, setup_style
from finrag.qa import load_answerable_questions

import matplotlib.pyplot as plt

CORRECT = 2  # judge correctness value treated as fully correct


def generation_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        s = summarize_generation(r)
        judge = primary_judge_name(r)
        j = s.get(f"judge:{judge}", {}) if judge else {}
        rows.append(
            {
                "label": r["label"],
                "model": r["config"]["gen_model"],
                "context": r["config"]["context_mode"],
                "k": r["config"]["k"],
                "accuracy": j.get("fully_correct_rate"),
                "mean_correctness": j.get("mean_correctness"),
                "numeric": s["numeric_accuracy"],
                "numeric_scale_tol": s["numeric_accuracy_scale_tolerant"],
                "token_f1": s["token_f1"],
                "abstain": s["abstention_rate"],
                "cited": s["citation_rate"],
                "cite_valid": s["citation_valid_rate"],
                "grounded": j.get("mean_groundedness"),
                "halluc": j.get("hallucination_rate"),
                "s_per_q": s["mean_seconds_per_question"],
                "truncations": s["n_truncation_suspected"],
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False, na_position="last")


def _correct_ids(run: dict) -> set[str]:
    judge = primary_judge_name(run)
    return {
        g["financebench_id"]
        for g in run["generations"]
        if g.get("verdicts", {}).get(judge, {}).get("correctness") == CORRECT
    }


def fig_gen_models(runs: list[dict]) -> None:
    retrieved = [r for r in runs if r["config"]["context_mode"] == "retrieved"]
    retrieved.sort(key=lambda r: summarize_generation(r)["numeric_accuracy_scale_tolerant"] or 0)
    if not retrieved:
        return
    models = [r["config"]["gen_model"].split(":")[0] + ":" + r["config"]["gen_model"].split(":")[1][:3] for r in retrieved]
    judge_acc, numeric = [], []
    for r in retrieved:
        s = summarize_generation(r)
        judge = primary_judge_name(r)
        judge_acc.append((s.get(f"judge:{judge}") or {}).get("fully_correct_rate") or 0)
        numeric.append(s["numeric_accuracy_scale_tolerant"] or 0)
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    b1 = ax.bar(x - 0.2, judge_acc, width=0.36, color=PALETTE[0], label="judge fully correct")
    b2 = ax.bar(x + 0.2, numeric, width=0.36, color=PALETTE[1], label="numeric (scale tolerant)")
    bar_labels(ax, b1)
    bar_labels(ax, b2)
    ax.set_xticks(x, models)
    ax.set_ylabel("Accuracy")
    ax.set_title("Generation models under retrieved context")
    ax.grid(axis="y")
    ax.legend(fontsize=8)
    save(fig, FIGURES / "fig_gen_models.png")


def fig_context_modes(runs: list[dict]) -> None:
    modes = ("none", "retrieved", "oracle")
    by_model: dict[str, dict[str, float]] = {}
    for r in runs:
        s = summarize_generation(r)
        judge = primary_judge_name(r)
        acc = (s.get(f"judge:{judge}") or {}).get("fully_correct_rate")
        if acc is None:  # closed book runs are unjudged; use numeric accuracy
            acc = s["numeric_accuracy_scale_tolerant"] or 0.0
        by_model.setdefault(r["config"]["gen_model"], {})[r["config"]["context_mode"]] = acc
    eligible = {m: v for m, v in by_model.items() if len(v) >= 2}
    if not eligible:
        return
    x = np.arange(len(modes))
    width = 0.8 / len(eligible)
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for i, (model, accs) in enumerate(sorted(eligible.items())):
        values = [accs.get(m, np.nan) for m in modes]
        bars = ax.bar(x + i * width, values, width=width * 0.9, color=PALETTE[i], label=model)
        bar_labels(ax, bars, fmt="{:.2f}")
    ax.set_xticks(x + 0.4 - width / 2, ["closed book", "retrieved", "oracle"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Context ablation: what bounds the system")
    ax.grid(axis="y")
    ax.legend(fontsize=8)
    save(fig, FIGURES / "fig_context_modes.png")


def fig_by_reasoning(runs: list[dict], questions) -> None:
    retrieved = [r for r in runs if r["config"]["context_mode"] == "retrieved"]
    if not retrieved:
        return
    best = max(
        retrieved,
        key=lambda r: (summarize_generation(r).get(f"judge:{primary_judge_name(r)}") or {}).get(
            "fully_correct_rate"
        )
        or 0,
    )
    correct = _correct_ids(best)
    reasoning = {q.financebench_id: (q.question_reasoning or "unspecified").split(" OR ")[0] for q in questions}
    counts = Counter(reasoning[g["financebench_id"]] for g in best["generations"])
    keep = [c for c, n in counts.most_common() if n >= 5]
    rates = []
    for cat in keep:
        ids = [g["financebench_id"] for g in best["generations"] if reasoning[g["financebench_id"]] == cat]
        rates.append(sum(1 for i in ids if i in correct) / len(ids))
    order = np.argsort(rates)
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    names = [f"{keep[i]} (n={counts[keep[i]]})" for i in order]
    bars = ax.barh(names, [rates[i] for i in order], color=PALETTE[0], height=0.62)
    for b, v in zip(bars, [rates[i] for i in order]):
        ax.annotate(f"{v:.2f}", (v, b.get_y() + b.get_height() / 2),
                    textcoords="offset points", xytext=(3, 0), va="center", fontsize=8, color=INK)
    ax.set_xlabel("Fully correct rate")
    ax.set_title(f"Accuracy by reasoning type ({best['label']})")
    ax.grid(axis="x")
    save(fig, FIGURES / "fig_by_reasoning.png")


def decomposition(runs: list[dict], questions) -> str | None:
    oracle = {r["config"]["gen_model"]: r for r in runs if r["config"]["context_mode"] == "oracle"}
    retrieved = {r["config"]["gen_model"]: r for r in runs if r["config"]["context_mode"] == "retrieved"}
    shared = set(oracle) & set(retrieved)
    if not shared:
        return None
    model = max(shared, key=lambda m: len(_correct_ids(oracle[m])))
    ret_ok, ora_ok = _correct_ids(retrieved[model]), _correct_ids(oracle[model])
    ids = [g["financebench_id"] for g in retrieved[model]["generations"]]
    solved = [i for i in ids if i in ret_ok and i in ora_ok]
    retrieval_limited = [i for i in ids if i not in ret_ok and i in ora_ok]
    generation_limited = [i for i in ids if i not in ora_ok]
    rescued = [i for i in ids if i in ret_ok and i not in ora_ok]
    n = len(ids)

    reasoning = {q.financebench_id: (q.question_reasoning or "unspecified").split(" OR ")[0] for q in questions}
    lines = [
        "# Retrieval versus generation: locating the bottleneck",
        "",
        f"Model: `{model}`, {n} questions, correctness judged by the primary judge.",
        "",
        "| Outcome | Count | Share | Meaning |",
        "|---|---|---|---|",
        f"| Solved | {len(solved)} | {len(solved)/n:.1%} | correct with retrieved and with gold evidence |",
        f"| Retrieval-limited | {len(retrieval_limited)} | {len(retrieval_limited)/n:.1%} | correct with gold evidence, wrong with retrieved |",
        f"| Generation-limited | {len(generation_limited)} | {len(generation_limited)/n:.1%} | wrong even with gold evidence |",
        f"| Retrieval-rescued | {len(rescued)} | {len(rescued)/n:.1%} | wrong with gold evidence, right with retrieved |",
        "",
        "The retrieval-limited share is the headroom a better retriever could",
        "recover; the generation-limited share is the ceiling of the model itself.",
        "A large retrieval-rescued share would suggest answers arriving from",
        "pre-training rather than the corpus and should be read against the",
        "closed-book run.",
        "",
        "## Generation-limited questions by reasoning type",
        "",
        "| Reasoning | Count |",
        "|---|---|",
    ]
    for cat, cnt in Counter(reasoning[i] for i in generation_limited).most_common():
        lines.append(f"| {cat} | {cnt} |")
    lines += ["", "## Retrieval-limited questions by reasoning type", "", "| Reasoning | Count |", "|---|---|"]
    for cat, cnt in Counter(reasoning[i] for i in retrieval_limited).most_common():
        lines.append(f"| {cat} | {cnt} |")
    return "\n".join(lines)


def main() -> None:
    setup_style()
    runs = load_generation_runs()
    if not runs:
        raise SystemExit("No generation runs found; run experiments/run_generation.py first.")
    questions, _ = load_answerable_questions()
    print(f"Analysing {len(runs)} generation runs")

    df = generation_table(runs)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "generation_results.csv", index=False)
    md = ["# Generation results (sorted by judged accuracy)", "", df.to_markdown(index=False, floatfmt=".3f")]
    (TABLES / "generation_results.md").write_text("\n".join(md))
    print(f"  generation_results.md / .csv ({len(df)} rows)")

    fig_gen_models(runs)
    fig_context_modes(runs)
    fig_by_reasoning(runs, questions)

    decomp = decomposition(runs, questions)
    if decomp:
        (TABLES / "retrieval_vs_generation.md").write_text(decomp)
        print("  retrieval_vs_generation.md")


if __name__ == "__main__":
    main()
