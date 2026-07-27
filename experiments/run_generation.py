"""Generation runs: the model sweep and the oracle and closed-book ablations.

Passages come from a stored retrieval run (by default the highest-recall run
whose k is large enough). Each generation is scored deterministically as it is
produced, and the primary judge (local, via Ollama) is applied afterwards; both
steps are idempotent and resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finrag.analysis import load_retrieval_runs, summarize_generation
from finrag.config import GENERATIONS_DIR, GenerationConfig
from finrag.experiment import passages_from_run
from finrag.generation import GenerationRun, oracle_passages
from finrag.io import read_json, write_json
from finrag.judge import build_judge_prompt, make_judge
from finrag.qa import load_answerable_questions

DEFAULT_MODELS = [
    "llama3.1:8b-instruct-q4_K_M",
    "mistral:7b-instruct",
    "qwen2.5:7b-instruct",
    "phi4:14b",
    "qwen2.5:14b-instruct-q4_K_M",
]
DEFAULT_JUDGE = "ollama:qwen2.5:14b-instruct-q4_K_M"


def pick_retrieval_run(run_id: str | None, k: int) -> dict:
    runs = load_retrieval_runs()
    if run_id:
        for r in runs:
            if r["run_id"] == run_id or r["label"] == run_id:
                return r
        raise SystemExit(f"No retrieval run named {run_id}")
    eligible = [r for r in runs if r["config"]["k"] >= k]
    if not eligible:
        raise SystemExit("No stored retrieval run has k large enough; run the grid first.")
    return eligible[0]


def judge_run(gen_path: Path, judge_spec: str, passages_for: dict, questions) -> None:
    """Attach verdicts from judge_spec to every unjudged generation in the file."""
    judge = make_judge(judge_spec)
    record = read_json(gen_path)
    by_id = {q.financebench_id: q for q in questions}
    todo = [
        g for g in record["generations"] if judge.name not in g.get("verdicts", {})
    ]
    if not todo:
        return
    print(f"Judging {len(todo)} answers with {judge.name}")
    for i, g in enumerate(todo, 1):
        q = by_id[g["financebench_id"]]
        passages = passages_for.get(q.financebench_id, [])
        k = record["config"]["k"] if record["config"]["context_mode"] != "none" else 0
        prompt = build_judge_prompt(
            q.question,
            q.answer,
            g["answer"],
            [p["text"] for p in passages[:k]],
            justification=q.justification,
        )
        verdict = judge.judge(prompt)
        g.setdefault("verdicts", {})[judge.name] = (
            verdict.to_dict() if verdict else {"parse_failure": True}
        )
        if i % 10 == 0 or i == len(todo):
            write_json(gen_path, record)
            print(f"  [{i}/{len(todo)}] judged")
    write_json(gen_path, record)


def run_one(
    gen_model: str,
    context_mode: str,
    retrieval_run: dict | None,
    k: int,
    label: str,
    judge_spec: str | None,
    questions,
) -> None:
    cfg = GenerationConfig(
        label=label,
        gen_model=gen_model,
        context_mode=context_mode,
        retrieval_run_id=retrieval_run["run_id"] if retrieval_run else "",
        k=k if context_mode == "retrieved" else (5 if context_mode == "oracle" else 0),
    )
    if context_mode == "retrieved":
        passages_for = passages_from_run(retrieval_run, questions)
    elif context_mode == "oracle":
        passages_for = {q.financebench_id: oracle_passages(q) for q in questions}
    else:
        passages_for = {}

    run = GenerationRun(cfg)
    run.run(questions, passages_for)
    if judge_spec and context_mode != "none":
        judge_run(run.path, judge_spec, passages_for, questions)

    record = read_json(run.path)
    print(f"\n{label}  [{gen_model}, context={context_mode}, k={cfg.k}]")
    print(json.dumps(summarize_generation(record), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-model", help="one Ollama model tag")
    parser.add_argument("--all", action="store_true", help="run the default five-model sweep")
    parser.add_argument("--context-mode", default="retrieved", choices=["retrieved", "oracle", "none"])
    parser.add_argument("--retrieval-run", help="run_id or label; default: best eligible recall")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--name", help="run label; default derived from model and mode")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help="judge spec or 'none'")
    args = parser.parse_args()

    questions, _ = load_answerable_questions()
    retrieval_run = None
    if args.context_mode == "retrieved":
        retrieval_run = pick_retrieval_run(args.retrieval_run, args.k)
        print(
            f"Retrieval run: {retrieval_run['label']} "
            f"(recall@{retrieval_run['config']['k']} = {retrieval_run['summary']['recall']:.3f})"
        )

    judge_spec = None if args.judge == "none" else args.judge
    models = DEFAULT_MODELS if args.all else [args.gen_model]
    if models == [None]:
        raise SystemExit("Pass --gen-model or --all")

    for model in models:
        label = args.name or f"{args.context_mode}:{model}"
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        run_one(
            model, args.context_mode, retrieval_run, args.k, label, judge_spec, questions
        )


if __name__ == "__main__":
    main()
