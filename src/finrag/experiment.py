"""Running and summarising retrieval configurations.

A retrieval run evaluates one RetrievalConfig over the whole evaluation set,
stores per-question metrics plus the retrieved chunk ids (texts are
reconstructable from the chunk cache, which keeps run files small enough to
commit), and caches the result under the config hash so sweeps are resumable.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from .chunking import build_chunks
from .config import RUNS_DIR, RetrievalConfig, ensure_dirs
from .embeddings import embed_texts, load_or_build_embeddings
from .io import read_json, write_json
from .metrics import retrieval_metrics
from .qa import Question

METRIC_NAMES = ("recall", "strict_recall", "doc_accuracy", "mrr", "ndcg", "evidence_coverage")


def run_retrieval_config(
    cfg: RetrievalConfig,
    questions: list[Question],
    device: str = "cpu",
    force: bool = False,
) -> dict:
    """Evaluate cfg on questions; loads the cached run if it already exists."""
    ensure_dirs()
    out_path = RUNS_DIR / f"{cfg.run_id()}.json"
    if out_path.exists() and not force:
        return read_json(out_path)

    from .retrieval import Retriever  # local import to keep module load light

    chunks = build_chunks(cfg)
    needs_dense = cfg.strategy in ("dense", "hybrid", "rerank")
    matrix = load_or_build_embeddings(cfg, chunks, device) if needs_dense else None
    retriever = Retriever(cfg, chunks, matrix, device=device)

    query_vecs = (
        embed_texts(
            [q.question for q in questions], cfg.embedding_model, device, is_query=True
        )
        if needs_dense
        else [None] * len(questions)
    )

    per_question = []
    t0 = time.time()
    for q, vec in zip(questions, query_vecs):
        retrieved = retriever.search(q.question, query_vec=vec)
        per_question.append(
            {
                "financebench_id": q.financebench_id,
                "question_type": q.question_type,
                "question_reasoning": q.question_reasoning,
                "metrics": retrieval_metrics(retrieved, q),
                "retrieved": [[c["chunk_id"], round(c["score"], 4)] for c in retrieved],
            }
        )
    ms_per_query = (time.time() - t0) / len(questions) * 1000

    record = {
        "label": cfg.label,
        "config": cfg.__dict__,
        "run_id": cfg.run_id(),
        "n_chunks": len(chunks),
        "ms_per_query": round(ms_per_query, 1),
        "summary": summarize_retrieval(per_question),
        "per_question": per_question,
    }
    write_json(out_path, record)
    return record


def summarize_retrieval(per_question: list[dict]) -> dict:
    """Mean metrics overall and split by question type."""
    summary: dict = {}
    for name in METRIC_NAMES:
        summary[name] = float(np.mean([p["metrics"][name] for p in per_question]))
    by_type: dict[str, list[dict]] = defaultdict(list)
    for p in per_question:
        by_type[p["question_type"]].append(p)
    summary["by_question_type"] = {
        t: {
            "n": len(items),
            **{
                name: float(np.mean([p["metrics"][name] for p in items]))
                for name in METRIC_NAMES
            },
        }
        for t, items in sorted(by_type.items())
    }
    return summary


def passages_from_run(run_record: dict, questions: list[Question]) -> dict[str, list[dict]]:
    """Reconstruct full passage dicts (with text) for a stored retrieval run."""
    cfg = RetrievalConfig(**run_record["config"])
    chunks = build_chunks(cfg)
    by_id = {c["chunk_id"]: c for c in chunks}
    wanted = {q.financebench_id for q in questions}
    out: dict[str, list[dict]] = {}
    for p in run_record["per_question"]:
        if p["financebench_id"] in wanted:
            out[p["financebench_id"]] = [by_id[cid] for cid, _ in p["retrieved"]]
    return out


def passages_for_generation(gen_record: dict, questions: list[Question]) -> dict[str, list[dict]]:
    """Passages a stored generation run saw, keyed by question id."""
    from .generation import oracle_passages  # avoid a module cycle

    mode = gen_record["config"]["context_mode"]
    if mode == "none":
        return {}
    if mode == "oracle":
        return {q.financebench_id: oracle_passages(q) for q in questions}
    run_path = RUNS_DIR / f"{gen_record['config']['retrieval_run_id']}.json"
    return passages_from_run(read_json(run_path), questions)
