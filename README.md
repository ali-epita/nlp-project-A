# finrag: Retrieval-Augmented Generation over FinanceBench

NLP Graded Project A. A complete RAG pipeline answering questions from a closed
collection of SEC filings, evaluated against the FinanceBench open source split.

Team: Marwane Mhenni, Hamza El hamdi, Ali Cherri.

## What the system does

1. Downloads the 84 source PDFs referenced by the 150 FinanceBench questions,
   falling back to the official FinanceBench GitHub mirror for rotted links,
   and records the provenance of every file in a manifest.
2. Extracts a page-level corpus (evidence in FinanceBench is annotated by page,
   so pages are the unit retrieval is scored against).
3. Chunks, embeds, and indexes the corpus, then sweeps the retrieval design
   space: chunk size and overlap, structure preservation, six embedding
   models, k, dense versus BM25 versus hybrid versus cross-encoder reranking,
   and metadata prefiltering parsed from the question text.
4. Generates answers with five free, locally executable instruction models
   served by Ollama, with inline citations of the passages used.
5. Evaluates retrieval deterministically (page recall, MRR, nDCG, evidence
   coverage) and generation with numeric accuracy, token F1, citation
   validity, and an LLM judge whose verdicts are themselves validated by
   cross-judge agreement.
6. Ablates the pipeline with oracle-context and closed-book runs to attribute
   errors to retrieval or generation.

## Repository layout

    src/finrag/          the package: corpus, chunking, retrieval, generation,
                         metrics, judges, analysis helpers
    experiments/         run_grid.py, run_generation.py, judge_agreement.py,
                         analyze_retrieval.py, analyze_generation.py
    tests/               unit tests (no network or GPU required)
    notebooks/           thin runner notebook mirroring the commands below
    scripts/             GPU pod setup
    data/                the two FinanceBench JSONL files (committed);
                         pdfs/, corpus/, cache/ are generated and gitignored
    results/             run records, tables, and figures, generated locally by
                         the sweep and analysis scripts and not committed
    report/              the scientific report (report_nlp.pdf)

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is
strongly recommended for the sweeps; the unit tests and single-question demo
run anywhere.

    uv sync
    uv run pytest

Generation additionally requires [Ollama](https://ollama.com) with the five
models pulled (see scripts/setup_pod.sh for the exact list).

## Reproducing the experiments

    # 1. Corpus (about 10 minutes, network bound)
    uv run finrag download
    uv run finrag extract

    # 2. Retrieval grid (about 45 configurations; minutes on an A100)
    uv run python experiments/run_grid.py --sweep all --device cuda
    uv run python experiments/analyze_retrieval.py

    # 3. Generation sweep and ablations (a few hours on an A100)
    uv run python experiments/run_generation.py --all -k 10
    uv run python experiments/run_generation.py --gen-model qwen2.5:14b-instruct-q4_K_M --context-mode oracle
    uv run python experiments/run_generation.py --gen-model qwen2.5:7b-instruct --context-mode oracle
    uv run python experiments/run_generation.py --gen-model qwen2.5:14b-instruct-q4_K_M --context-mode none --judge none
    uv run python experiments/run_generation.py --gen-model qwen2.5:7b-instruct --context-mode none --judge none
    uv run python experiments/analyze_generation.py

    # 4. Judge validation (requires the Codex CLI for the secondary judge)
    uv run python experiments/judge_agreement.py --sample 60

The built report is committed at report/report_nlp.pdf. Its source is not
tracked, so the make report target does not run from a fresh clone.

Every stage is cached and resumable: retrieval runs are keyed by a hash of
their configuration, generation runs checkpoint every ten questions, and
re-running any command skips completed work. To answer a single question
interactively:

    uv run finrag ask "What was 3M's FY2018 capital expenditure?"

## Notes on reproducibility

About one fifth of the catalog's doc_link URLs are dead; the downloader's
GitHub fallback restores full coverage, and data/pdfs/manifest.json records
which source supplied each document. Questions whose document nevertheless
fails to download are excluded from the evaluation set and counted in the
report. All randomness in evaluation sampling is seeded.
