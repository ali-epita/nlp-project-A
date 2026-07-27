"""Judge validation by cross-judge agreement.

The primary (local) judge's verdicts are validated by re-judging a fixed random
sample of answers with an independent, stronger judge (gpt-5.5 through the
Codex CLI) and reporting exact agreement plus Cohen's kappa on the ordinal
correctness scale. The sample is seeded, so re-running reproduces the same
selection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finrag.analysis import primary_judge_name
from finrag.config import GENERATIONS_DIR, RESULTS
from finrag.experiment import passages_for_generation
from finrag.io import read_json, write_json
from finrag.judge import build_judge_prompt, cohen_kappa, make_judge
from finrag.qa import load_answerable_questions


def pick_run(label: str | None) -> tuple[dict, Path]:
    candidates = []
    for p in sorted(GENERATIONS_DIR.glob("*.json")):
        r = read_json(p)
        if r["config"]["context_mode"] == "retrieved":
            candidates.append((r, p))
    if label:
        for r, p in candidates:
            if r["label"] == label:
                return r, p
        raise SystemExit(f"No retrieved generation run labelled {label}")
    judged = [(r, p) for r, p in candidates if primary_judge_name(r)]
    if not judged:
        raise SystemExit("No judged retrieved run found; run the generation sweep first.")
    return judged[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="generation run label; default: first judged retrieved run")
    parser.add_argument("--secondary", default="codex", help="secondary judge spec")
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    record, path = pick_run(args.run)
    primary = primary_judge_name(record)
    judged = [g for g in record["generations"] if primary in g.get("verdicts", {})]
    rng = random.Random(args.seed)
    sample = rng.sample(judged, min(args.sample, len(judged)))
    print(f"Run: {record['label']} | primary judge: {primary} | sample: {len(sample)}")

    questions, _ = load_answerable_questions()
    by_id = {q.financebench_id: q for q in questions}
    passages_for = passages_for_generation(record, questions)
    secondary = make_judge(args.secondary)

    k = record["config"]["k"]
    for i, g in enumerate(sample, 1):
        if secondary.name in g.get("verdicts", {}):
            continue
        q = by_id[g["financebench_id"]]
        prompt = build_judge_prompt(
            q.question,
            q.answer,
            g["answer"],
            [p["text"] for p in passages_for.get(q.financebench_id, [])[:k]],
            justification=q.justification,
        )
        verdict = secondary.judge(prompt)
        g.setdefault("verdicts", {})[secondary.name] = (
            verdict.to_dict() if verdict else {"parse_failure": True}
        )
        if i % 5 == 0 or i == len(sample):
            write_json(path, record)
            print(f"  [{i}/{len(sample)}] judged by {secondary.name}")
    write_json(path, record)

    pairs = [
        (g["verdicts"][primary]["correctness"], g["verdicts"][secondary.name]["correctness"])
        for g in sample
        if "correctness" in g["verdicts"].get(primary, {})
        and "correctness" in g["verdicts"].get(secondary.name, {})
    ]
    if len(pairs) < 2:
        raise SystemExit("Not enough paired verdicts to compute agreement.")
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    result = {
        "run": record["label"],
        "primary_judge": primary,
        "secondary_judge": secondary.name,
        "n_paired": len(pairs),
        "exact_agreement": sum(x == y for x, y in pairs) / len(pairs),
        "kappa_unweighted": cohen_kappa(a, b),
        "kappa_linear_weighted": cohen_kappa(a, b, weighted=True),
        "mean_correctness_primary": sum(a) / len(a),
        "mean_correctness_secondary": sum(b) / len(b),
    }
    write_json(RESULTS / "judge_agreement.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
