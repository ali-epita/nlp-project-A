"""Text extraction from the downloaded PDFs.

Extraction is page based: FinanceBench evidence is annotated with zero-based
page numbers, so keeping the page as the atomic unit lets retrieval be scored
against the gold pages directly. Cleaning is deliberately conservative; heavy
normalisation of financial tables destroys the row and column adjacency that
retrieval later depends on.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF

from ..config import CORPUS_DIR, PAGES_PATH, PDF_DIR, ensure_dirs
from ..io import write_json, write_jsonl

# Pages with fewer characters than this are likely scanned images or covers.
LOW_TEXT_THRESHOLD = 20


def clean_text(text: str) -> str:
    """Normalise unicode, strip control characters, and bound whitespace runs."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if c == "\n" or c == "\t" or unicodedata.category(c)[0] != "C")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_document(pdf_path: Path) -> list[dict]:
    """Extract one PDF into a list of page records."""
    doc_name = pdf_path.stem
    pages: list[dict] = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            text = clean_text(page.get_text("text"))
            pages.append(
                {
                    "doc_name": doc_name,
                    "page_num": page_num,
                    "text": text,
                    "n_chars": len(text),
                }
            )
    return pages


def extract_corpus(pdf_dir: Path = PDF_DIR) -> dict:
    """Extract every PDF in pdf_dir into pages.jsonl plus a stats file."""
    ensure_dirs()
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    all_pages: list[dict] = []
    per_doc: list[dict] = []
    for i, path in enumerate(pdf_paths, 1):
        pages = extract_document(path)
        all_pages.extend(pages)
        low_text = sum(1 for p in pages if p["n_chars"] < LOW_TEXT_THRESHOLD)
        per_doc.append(
            {
                "doc_name": path.stem,
                "n_pages": len(pages),
                "n_chars": sum(p["n_chars"] for p in pages),
                "n_low_text_pages": low_text,
            }
        )
        print(f"[{i:3d}/{len(pdf_paths)}] {path.stem:45s} {len(pages):4d} pages")

    write_jsonl(PAGES_PATH, all_pages)
    stats = {
        "n_documents": len(per_doc),
        "n_pages": len(all_pages),
        "n_chars": sum(p["n_chars"] for p in all_pages),
        "n_low_text_pages": sum(d["n_low_text_pages"] for d in per_doc),
        "documents": per_doc,
    }
    write_json(CORPUS_DIR / "corpus_stats.json", stats)
    print(
        f"\n{stats['n_pages']:,} pages, {stats['n_chars']:,} characters "
        f"from {stats['n_documents']} documents -> {PAGES_PATH}"
    )
    return stats
