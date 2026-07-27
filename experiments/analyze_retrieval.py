"""Analysis of the retrieval grid: tables, figures, and the error analysis.

Sweep membership is resolved through the grid definition itself (run ids, not
label parsing), so renaming a label cannot silently detach a run from its
sweep.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from finrag.analysis import load_retrieval_runs
from finrag.config import FIGURES, TABLES
from finrag.io import read_json, write_json
from finrag.plotting import INK, PALETTE, SEQUENTIAL_CMAP, bar_labels, save, setup_style
from finrag.qa import load_answerable_questions
from run_grid import build_grid

import matplotlib.pyplot as plt


def results_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        c, s = r["config"], r["summary"]
        rows.append(
            {
                "label": r["label"],
                "strategy": c["strategy"],
                "chunking": f"{c['chunker']} {c['chunk_size']}/{c['chunk_overlap']}",
                "embedding": c["embedding_model"].split("/")[-1],
                "k": c["k"],
                "filter": "yes" if c["metadata_filter"] else "",
                "recall": s["recall"],
                "strict_recall": s["strict_recall"],
                "doc_accuracy": s["doc_accuracy"],
                "mrr": s["mrr"],
                "ndcg": s["ndcg"],
                "evidence_coverage": s["evidence_coverage"],
                "ms_per_query": r["ms_per_query"],
            }
        )
    return pd.DataFrame(rows).sort_values("recall", ascending=False)


def runs_by_id(runs: list[dict]) -> dict[str, dict]:
    return {r["run_id"]: r for r in runs}


def sweep_runs(sweep_name: str, by_id: dict[str, dict]) -> list[tuple[object, dict]]:
    """(config, run) pairs for one sweep, in grid order, skipping unfinished runs."""
    grid = build_grid()
    out = []
    for cfg in grid[sweep_name]:
        run = by_id.get(cfg.run_id())
        if run is not None:
            out.append((cfg, run))
    return out


def fig_chunking(by_id: dict) -> None:
    pairs = sweep_runs("chunking", by_id)
    sizes = sorted({c.chunk_size for c, _ in pairs})
    pcts = sorted({round(c.chunk_overlap / c.chunk_size * 100) for c, _ in pairs})
    grid = np.full((len(sizes), len(pcts)), np.nan)
    for c, r in pairs:
        i = sizes.index(c.chunk_size)
        j = pcts.index(round(c.chunk_overlap / c.chunk_size * 100))
        grid[i, j] = r["summary"]["recall"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    im = ax.imshow(grid, cmap=SEQUENTIAL_CMAP, aspect="auto")
    ax.set_xticks(range(len(pcts)), [f"{p}%" for p in pcts])
    ax.set_yticks(range(len(sizes)), sizes)
    ax.set_xlabel("Chunk overlap (fraction of size)")
    ax.set_ylabel("Chunk size (tokens)")
    ax.set_title("Recall@5 by chunk size and overlap")
    ax.grid(False)
    mid = (np.nanmax(grid) + np.nanmin(grid)) / 2
    for i in range(len(sizes)):
        for j in range(len(pcts)):
            if not np.isnan(grid[i, j]):
                ax.text(
                    j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if grid[i, j] > mid else INK,
                )
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, FIGURES / "fig_chunking.png")


def _bar_figure(items: list[tuple[str, float]], title: str, path_name: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    labels = [n for n, _ in items]
    values = [v for _, v in items]
    bars = ax.bar(labels, values, color=color, width=0.62)
    bar_labels(ax, bars)
    ax.set_ylabel("Recall@5")
    ax.set_title(title)
    ax.grid(axis="y")
    ax.tick_params(axis="x", rotation=15)
    save(fig, FIGURES / path_name)


def fig_structure(by_id: dict) -> None:
    items = [(c.chunker, r["summary"]["recall"]) for c, r in sweep_runs("structure", by_id)]
    _bar_figure(items, "Structure preservation (512-token budget)", "fig_structure.png", PALETTE[0])


def fig_embeddings(by_id: dict) -> None:
    pairs = sweep_runs("embeddings", by_id)
    pairs.sort(key=lambda p: p[1]["summary"]["recall"])
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    names = [c.embedding_model.split("/")[-1] for c, _ in pairs]
    values = [r["summary"]["recall"] for _, r in pairs]
    bars = ax.barh(names, values, color=PALETTE[0], height=0.62)
    for b, v in zip(bars, values):
        ax.annotate(
            f"{v:.3f}", (v, b.get_y() + b.get_height() / 2),
            textcoords="offset points", xytext=(3, 0), va="center", fontsize=8, color=INK,
        )
    ax.set_xlabel("Recall@5")
    ax.set_title("Embedding models (identical chunks)")
    ax.grid(axis="x")
    save(fig, FIGURES / "fig_embeddings.png")


def fig_k(by_id: dict) -> None:
    pairs = sweep_runs("k", by_id)
    ks = [c.k for c, _ in pairs]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, metric in enumerate(("recall", "strict_recall", "evidence_coverage", "mrr")):
        values = [r["summary"][metric] for _, r in pairs]
        ax.plot(ks, values, marker="o", markersize=5, linewidth=2, color=PALETTE[i], label=metric)
        ax.annotate(
            metric, (ks[-1], values[-1]), textcoords="offset points", xytext=(6, 0),
            fontsize=8, color=INK, va="center",
        )
    ax.set_xticks(ks)
    ax.set_xlabel("k (passages retrieved)")
    ax.set_ylabel("Score")
    ax.set_title("Effect of k (dense, 512/64, bge-base)")
    ax.set_xlim(0, max(ks) * 1.25)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, FIGURES / "fig_k.png")


def fig_strategy(by_id: dict, runs: list[dict]) -> None:
    """Strategies compared at k=10 with the baseline chunks and embeddings."""
    wanted: list[tuple[str, dict]] = []
    for label_prefix, sweep in (("bm25", "strategy"), ("hybrid", "strategy"), ("rerank", "strategy")):
        candidates = [
            (c, r)
            for c, r in sweep_runs(sweep, by_id)
            if c.strategy == label_prefix and c.k == 10
        ]
        if candidates:
            best = max(candidates, key=lambda p: p[1]["summary"]["recall"])
            wanted.append((best[0].label, best[1]))
    dense10 = [(c, r) for c, r in sweep_runs("k", by_id) if c.k == 10]
    if dense10:
        wanted.insert(0, ("dense@k=10", dense10[0][1]))
    filt = [(c, r) for c, r in sweep_runs("filter", by_id) if c.k == 10 and c.strategy == "dense"]
    if filt:
        wanted.append(("dense+filter@k=10", filt[0][1]))

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels = [n for n, _ in wanted]
    values = [r["summary"]["recall"] for _, r in wanted]
    bars = ax.bar(labels, values, color=PALETTE[: len(wanted)], width=0.62)
    bar_labels(ax, bars)
    ax.set_ylabel("Recall@10")
    ax.set_title("Retrieval strategies at k=10 (best variant of each)")
    ax.grid(axis="y")
    ax.tick_params(axis="x", rotation=12)
    save(fig, FIGURES / "fig_strategy.png")

    # Per question type, same strategies, colours follow the strategy.
    types = sorted(wanted[0][1]["summary"]["by_question_type"]) if wanted else []
    if types:
        width = 0.8 / len(wanted)
        x = np.arange(len(types))
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for i, (name, r) in enumerate(wanted):
            values = [r["summary"]["by_question_type"][t]["recall"] for t in types]
            ax.bar(x + i * width, values, width=width * 0.92, color=PALETTE[i], label=name)
        ax.set_xticks(x + 0.4 - width / 2, types)
        ax.set_ylabel("Recall@10")
        ax.set_title("Strategies by question type")
        ax.grid(axis="y")
        ax.legend(fontsize=8)
        save(fig, FIGURES / "fig_strategy_by_type.png")


def error_analysis(runs: list[dict]) -> str:
    """Failure taxonomy for the best run at k<=10 (small k keeps ranking honest)."""
    questions, _ = load_answerable_questions()
    by_qid = {q.financebench_id: q for q in questions}
    eligible = [r for r in runs if r["config"]["k"] <= 10]
    best = max(eligible, key=lambda r: r["summary"]["recall"])
    k = best["config"]["k"]

    wrong_doc, wrong_page, boundary_loss = [], [], []
    for p in best["per_question"]:
        m = p["metrics"]
        if m["recall"] == 0:
            (wrong_page if m["doc_accuracy"] else wrong_doc).append(p)
        elif m["evidence_coverage"] < 0.2:
            boundary_loss.append(p)
    failures = wrong_doc + wrong_page
    n = len(best["per_question"])

    lines = [
        "# Retrieval Error Analysis",
        "",
        f"Best configuration with k <= 10: **{best['label']}** "
        f"(recall@{k} = {best['summary']['recall']:.3f} over {n} questions)",
        "",
        "## Failure taxonomy",
        "",
        "| Failure mode | Count | Share of questions |",
        "|---|---|---|",
        f"| Wrong document retrieved | {len(wrong_doc)} | {len(wrong_doc)/n:.1%} |",
        f"| Right document, wrong page | {len(wrong_page)} | {len(wrong_page)/n:.1%} |",
        f"| Gold page hit but evidence text mostly absent | {len(boundary_loss)} | {len(boundary_loss)/n:.1%} |",
        f"| **Total misses at k={k}** | **{len(failures)}** | {len(failures)/n:.1%} |",
        "",
        "Wrong-document failures are discrimination errors (company, year, or filing",
        "type not pinned down) and respond to metadata filtering. Right-document",
        "wrong-page failures are ranking errors inside one filing and respond to",
        "chunking and reranking. The third row only shows up in text-level measures:",
        "the page was retrieved but the chunk boundary cut the evidence away.",
        "",
        "## Failures by question type",
        "",
        "| Question type | Failures | Total | Failure rate |",
        "|---|---|---|---|",
    ]
    for qtype in sorted({p["question_type"] for p in best["per_question"]}):
        total = sum(1 for p in best["per_question"] if p["question_type"] == qtype)
        fail = sum(1 for p in failures if p["question_type"] == qtype)
        lines.append(f"| {qtype} | {fail} | {total} | {fail/total:.1%} |")

    lines += ["", "## Individual failures", "", "| id | type | doc found | coverage |", "|---|---|---|---|"]
    for p in failures:
        lines.append(
            f"| {p['financebench_id']} | {p['question_type']} "
            f"| {'yes' if p['metrics']['doc_accuracy'] else 'no'} "
            f"| {p['metrics']['evidence_coverage']:.2f} |"
        )
    lines += [
        "",
        "## Questions excluded by the metadata filter analysis",
        "",
        f"Companies parsed from question text cover the filter runs; see the",
        f"strategy figure for the filter's effect at matched k.",
    ]
    return "\n".join(lines)


def main() -> None:
    setup_style()
    runs = load_retrieval_runs()
    if not runs:
        raise SystemExit("No retrieval runs found; run experiments/run_grid.py first.")
    print(f"Analysing {len(runs)} retrieval runs")

    df = results_table(runs)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "retrieval_results.csv", index=False)
    md = ["# Retrieval results (all configurations, sorted by recall)", "", df.to_markdown(index=False, floatfmt=".3f")]
    (TABLES / "retrieval_results.md").write_text("\n".join(md))
    print(f"  retrieval_results.md / .csv ({len(df)} rows)")

    by_id = runs_by_id(runs)
    fig_chunking(by_id)
    fig_structure(by_id)
    fig_embeddings(by_id)
    fig_k(by_id)
    fig_strategy(by_id, runs)

    (TABLES / "error_analysis.md").write_text(error_analysis(runs))
    print("  error_analysis.md")


if __name__ == "__main__":
    main()
