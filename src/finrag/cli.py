"""Command line entry points: corpus building and an interactive demo.

The experiment sweeps live in experiments/ as standalone scripts; this CLI
covers the pipeline steps a user runs once (download, extract) and a small
demo that answers a single question with citations, which is the system in its
intended form.
"""

from __future__ import annotations

import argparse
import sys

from .config import RetrievalConfig


def cmd_download(args: argparse.Namespace) -> None:
    from .data.download import download_corpus

    download_corpus(only_needed=not args.all)


def cmd_extract(args: argparse.Namespace) -> None:
    from .data.extract import extract_corpus

    extract_corpus()


def cmd_ask(args: argparse.Namespace) -> None:
    from .chunking import build_chunks
    from .embeddings import load_or_build_embeddings, resolve_device
    from .generation import build_prompt, format_passages, ollama_chat
    from .retrieval import Retriever

    cfg = RetrievalConfig(
        strategy=args.strategy, k=args.k, metadata_filter=not args.no_filter
    )
    device = resolve_device("auto")
    chunks = build_chunks(cfg)
    matrix = load_or_build_embeddings(cfg, chunks, device)
    retriever = Retriever(cfg, chunks, matrix, device=device)
    passages = retriever.search(args.question)

    print("\nRetrieved passages:")
    for i, p in enumerate(passages, 1):
        pages = f"p.{p['page_start']}" if p["page_start"] == p["page_end"] else f"pp.{p['page_start']}-{p['page_end']}"
        print(f"  [{i}] {p['doc_name']} {pages} (score {p['score']:.3f})")

    system, user = build_prompt(args.question, passages, "retrieved")
    result = ollama_chat(args.model, system, user)
    print(f"\nAnswer ({args.model}):\n{result['text']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="finrag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download the FinanceBench PDFs")
    p.add_argument("--all", action="store_true", help="all 361 catalog documents, not only the 84 needed")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("extract", help="extract PDFs into the page corpus")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("ask", help="answer one question with the RAG pipeline")
    p.add_argument("question")
    p.add_argument("--model", default="qwen2.5:7b-instruct")
    p.add_argument("--strategy", default="dense")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--no-filter", action="store_true")
    p.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
