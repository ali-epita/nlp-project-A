"""Document segmentation strategies.

Three chunkers cover the structure-preservation axis of the experiments:

- token: fixed-size sliding windows over the whole document, ignoring layout.
- page: the PDF page is the atomic unit; oversized pages are split internally.
- structural: paragraphs are packed greedily up to the token budget so chunk
  boundaries never cut through a paragraph.

All strategies measure size in tokens of a single reference tokenizer
(config.CHUNK_TOKENIZER). Chunk boundaries therefore stay identical when the
embedding model changes, which keeps the embedding comparison controlled.

A chunk record carries the doc name and the zero-based page span it covers, so
retrieval output can be scored against FinanceBench's page-annotated evidence.
"""

from __future__ import annotations

import functools
from collections import defaultdict

from transformers import AutoTokenizer

from .config import CACHE_DIR, PAGES_PATH, RetrievalConfig, CHUNK_TOKENIZER, ensure_dirs
from .io import read_jsonl, write_jsonl

PAGE_BREAK = "\n\n"


@functools.lru_cache(maxsize=1)
def get_tokenizer():
    # model_max_length is lifted because whole documents pass through the
    # tokenizer for boundary computation; no model ever sees these sequences.
    return AutoTokenizer.from_pretrained(
        CHUNK_TOKENIZER, use_fast=True, model_max_length=10**9
    )


def count_tokens(text: str) -> int:
    return len(get_tokenizer()(text, add_special_tokens=False)["input_ids"])


def _doc_text_and_bounds(pages: list[dict]) -> tuple[str, list[tuple[int, int, int]]]:
    """Join a document's pages; return the text and (char_start, char_end, page_num) bounds."""
    parts: list[str] = []
    bounds: list[tuple[int, int, int]] = []
    pos = 0
    for p in pages:
        text = p["text"]
        parts.append(text)
        bounds.append((pos, pos + len(text), p["page_num"]))
        pos += len(text) + len(PAGE_BREAK)
    return PAGE_BREAK.join(parts), bounds


def _pages_for_span(bounds: list[tuple[int, int, int]], start: int, end: int) -> tuple[int, int]:
    covered = [pn for s, e, pn in bounds if s < end and e > start]
    if not covered:  # span falls entirely inside a page break
        covered = [min(bounds, key=lambda b: abs(b[0] - start))[2]]
    return min(covered), max(covered)


def _token_windows(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Character spans of fixed-size token windows with the given overlap."""
    enc = get_tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if not offsets:
        return []
    stride = max(size - overlap, 1)
    spans: list[tuple[int, int]] = []
    for start_tok in range(0, len(offsets), stride):
        window = offsets[start_tok : start_tok + size]
        spans.append((window[0][0], window[-1][1]))
        if start_tok + size >= len(offsets):
            break
    return spans


def _chunk_token(pages: list[dict], size: int, overlap: int) -> list[dict]:
    text, bounds = _doc_text_and_bounds(pages)
    chunks = []
    for cs, ce in _token_windows(text, size, overlap):
        p_start, p_end = _pages_for_span(bounds, cs, ce)
        chunks.append({"text": text[cs:ce], "page_start": p_start, "page_end": p_end})
    return chunks


def _chunk_page(pages: list[dict], size: int, overlap: int) -> list[dict]:
    chunks = []
    for p in pages:
        if not p["text"]:
            continue
        for cs, ce in _token_windows(p["text"], size, overlap):
            chunks.append(
                {"text": p["text"][cs:ce], "page_start": p["page_num"], "page_end": p["page_num"]}
            )
    return chunks


def _chunk_structural(pages: list[dict], size: int, overlap: int) -> list[dict]:
    """Pack whole paragraphs up to the token budget; split only oversized ones."""
    text, bounds = _doc_text_and_bounds(pages)
    paragraphs: list[tuple[int, int]] = []
    pos = 0
    for para in text.split("\n\n"):
        if para.strip():
            paragraphs.append((pos, pos + len(para)))
        pos += len(para) + 2

    chunks: list[dict] = []
    current: list[tuple[int, int]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            cs, ce = current[0][0], current[-1][1]
            p_start, p_end = _pages_for_span(bounds, cs, ce)
            chunks.append({"text": text[cs:ce], "page_start": p_start, "page_end": p_end})
            current, current_tokens = [], 0

    for ps, pe in paragraphs:
        n = count_tokens(text[ps:pe])
        if n > size:  # oversized paragraph: flush and token-split it
            flush()
            for cs, ce in _token_windows(text[ps:pe], size, overlap):
                p_start, p_end = _pages_for_span(bounds, ps + cs, ps + ce)
                chunks.append(
                    {"text": text[ps + cs : ps + ce], "page_start": p_start, "page_end": p_end}
                )
            continue
        if current_tokens + n > size:
            flush()
        current.append((ps, pe))
        current_tokens += n
    flush()
    return chunks


_CHUNKERS = {"token": _chunk_token, "page": _chunk_page, "structural": _chunk_structural}


def build_chunks(cfg: RetrievalConfig, pages: list[dict] | None = None) -> list[dict]:
    """Chunk the corpus for cfg, using the on-disk cache when available."""
    ensure_dirs()
    cache_path = CACHE_DIR / f"chunks_{cfg.chunk_key()}.jsonl"
    if cache_path.exists():
        return read_jsonl(cache_path)

    if pages is None:
        pages = read_jsonl(PAGES_PATH)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for p in pages:
        by_doc[p["doc_name"]].append(p)

    chunker = _CHUNKERS[cfg.chunker]
    chunks: list[dict] = []
    for doc_name in sorted(by_doc):
        doc_pages = sorted(by_doc[doc_name], key=lambda p: p["page_num"])
        for c in chunker(doc_pages, cfg.chunk_size, cfg.chunk_overlap):
            c["doc_name"] = doc_name
            c["chunk_id"] = len(chunks)
            chunks.append(c)

    write_jsonl(cache_path, chunks)
    return chunks
