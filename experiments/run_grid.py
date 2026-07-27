"""The retrieval experiment grid.

Sweeps are one-factor-at-a-time around a fixed baseline (token chunks of 512
with 64 overlap, bge-base embeddings, dense search, k=5) so each comparison
isolates a single design decision:

- chunking: chunk size x relative overlap (4 x 4)
- structure: token vs page vs structural segmentation
- embeddings: six models spanning parameter count and training recipe
- k: how many passages the generator would receive
- strategy: dense vs BM25 vs hybrid (three fusion weights) vs cross-encoder
  reranking (three candidate pools)
- filter: metadata prefiltering by company and period parsed from the question

Runs are cached by config hash: re-running the script only executes what is
missing, so the grid is resumable after any interruption.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finrag.config import RetrievalConfig
from finrag.embeddings import describe_device, resolve_device
from finrag.experiment import METRIC_NAMES, run_retrieval_config
from finrag.qa import load_answerable_questions

BASE = dict(
    strategy="dense",
    chunker="token",
    chunk_size=512,
    chunk_overlap=64,
    embedding_model="BAAI/bge-base-en-v1.5",
    k=5,
)

EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "thenlper/gte-base",
    "intfloat/e5-base-v2",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-large-v2",
    "BAAI/bge-large-en-v1.5",
]


def build_grid() -> dict[str, list[RetrievalConfig]]:
    sweeps: dict[str, list[RetrievalConfig]] = {}

    sweeps["chunking"] = [
        RetrievalConfig(
            **{
                **BASE,
                "label": f"chunk{size}/ov{pct}%",
                "chunk_size": size,
                "chunk_overlap": int(size * pct / 100),
            }
        )
        for size in (128, 256, 512, 1024)
        for pct in (0, 10, 25, 50)
    ]

    sweeps["structure"] = [
        RetrievalConfig(**{**BASE, "label": f"struct:{chunker}", "chunker": chunker})
        for chunker in ("token", "page", "structural")
    ]

    sweeps["embeddings"] = [
        RetrievalConfig(**{**BASE, "label": f"embed:{m.split('/')[-1]}", "embedding_model": m})
        for m in EMBEDDING_MODELS
    ]

    sweeps["k"] = [
        RetrievalConfig(**{**BASE, "label": f"k={k}", "k": k}) for k in (1, 3, 5, 10, 20)
    ]

    sweeps["strategy"] = (
        [
            RetrievalConfig(**{**BASE, "label": f"bm25@k={k}", "strategy": "bm25", "k": k})
            for k in (5, 10)
        ]
        + [
            RetrievalConfig(
                **{**BASE, "label": f"hybrid:a={a}@k=10", "strategy": "hybrid", "alpha": a, "k": 10}
            )
            for a in (0.3, 0.5, 0.7)
        ]
        + [
            RetrievalConfig(
                **{
                    **BASE,
                    "label": f"rerank:pool={pool}@k=10",
                    "strategy": "rerank",
                    "rerank_pool": pool,
                    "k": 10,
                }
            )
            for pool in (20, 50, 100)
        ]
        + [
            RetrievalConfig(
                **{
                    **BASE,
                    "label": "rerank:pool=100@k=20",
                    "strategy": "rerank",
                    "rerank_pool": 100,
                    "k": 20,
                }
            )
        ]
    )

    sweeps["filter"] = [
        RetrievalConfig(**{**BASE, "label": "filter@k=5", "metadata_filter": True}),
        RetrievalConfig(**{**BASE, "label": "filter@k=10", "metadata_filter": True, "k": 10}),
        RetrievalConfig(**{**BASE, "label": "filter@k=20", "metadata_filter": True, "k": 20}),
        RetrievalConfig(
            **{
                **BASE,
                "label": "filter+rerank:pool=100@k=10",
                "metadata_filter": True,
                "strategy": "rerank",
                "rerank_pool": 100,
                "k": 10,
            }
        ),
    ]
    return sweeps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", nargs="+", default=["all"], help="sweep names or 'all'")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true", help="recompute cached runs")
    args = parser.parse_args()

    sweeps = build_grid()
    selected = list(sweeps) if args.sweep == ["all"] else args.sweep
    configs: list[RetrievalConfig] = []
    seen: set[str] = set()
    for name in selected:
        for cfg in sweeps[name]:
            if cfg.run_id() not in seen:  # sweeps share their baseline point
                seen.add(cfg.run_id())
                configs.append(cfg)

    device = resolve_device(args.device)
    questions, missing = load_answerable_questions()
    print(f"Grid: {len(configs)} configurations across sweeps {selected}")
    print(f"Eval set: {len(questions)} answerable questions ({len(missing)} excluded)")
    print(f"Device: {describe_device(device)}\n")

    for i, cfg in enumerate(configs, 1):
        print(f"[{i:2d}/{len(configs)}] {cfg.label}")
        record = run_retrieval_config(cfg, questions, device=device, force=args.force)
        s = record["summary"]
        line = "  ".join(f"{m}={s[m]:.3f}" for m in METRIC_NAMES)
        print(f"  {record['n_chunks']:,} chunks | {record['ms_per_query']} ms/query | {line}\n")


if __name__ == "__main__":
    main()
