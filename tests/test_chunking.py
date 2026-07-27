"""Tests for the three chunking strategies on a synthetic document."""

import pytest

import finrag.chunking as chunking
from finrag.chunking import _chunk_page, _chunk_structural, _chunk_token, count_tokens
from finrag.config import RetrievalConfig


def pages(texts: list[str], doc="DOC_2020_10K") -> list[dict]:
    return [
        {"doc_name": doc, "page_num": i, "text": t, "n_chars": len(t)}
        for i, t in enumerate(texts)
    ]


LONG_PAGE = " ".join(f"word{i}" for i in range(300))
SHORT_PAGE = "short page about revenue"


class TestTokenChunker:
    def test_windows_cover_document(self):
        chunks = _chunk_token(pages([LONG_PAGE, SHORT_PAGE]), size=128, overlap=0)
        assert len(chunks) >= 3
        joined = " ".join(c["text"] for c in chunks)
        assert "word0" in joined and "word299" in joined and "revenue" in joined

    def test_overlap_repeats_content(self):
        no_ov = _chunk_token(pages([LONG_PAGE]), size=128, overlap=0)
        with_ov = _chunk_token(pages([LONG_PAGE]), size=128, overlap=32)
        assert len(with_ov) > len(no_ov)

    def test_page_span_tracked(self):
        chunks = _chunk_token(pages([LONG_PAGE, SHORT_PAGE]), size=4096, overlap=0)
        assert chunks[0]["page_start"] == 0 and chunks[0]["page_end"] == 1

    def test_chunk_size_respected(self):
        for c in _chunk_token(pages([LONG_PAGE]), size=128, overlap=0):
            assert count_tokens(c["text"]) <= 130  # small slack for offset rounding


class TestPageChunker:
    def test_chunks_never_cross_pages(self):
        chunks = _chunk_page(pages([LONG_PAGE, SHORT_PAGE]), size=128, overlap=0)
        assert all(c["page_start"] == c["page_end"] for c in chunks)

    def test_empty_pages_skipped(self):
        chunks = _chunk_page(pages(["", SHORT_PAGE]), size=128, overlap=0)
        assert all(c["text"] for c in chunks)
        assert chunks[0]["page_start"] == 1


class TestStructuralChunker:
    def test_paragraphs_not_split(self):
        paras = ["First paragraph about assets.", "Second paragraph about liabilities.",
                 "Third paragraph about equity."]
        chunks = _chunk_structural(pages(["\n\n".join(paras)]), size=512, overlap=0)
        assert len(chunks) == 1  # everything fits one budget
        for p in paras:
            assert p in chunks[0]["text"]

    def test_budget_forces_split_at_boundary(self):
        # Common words tokenize one-to-one, keeping each paragraph under budget
        # on its own but both together over it.
        paras = [" ".join(["revenue"] * 100), " ".join(["asset"] * 100)]
        chunks = _chunk_structural(pages(["\n\n".join(paras)]), size=150, overlap=0)
        assert len(chunks) == 2
        assert "revenue" in chunks[0]["text"] and "asset" in chunks[1]["text"]
        assert "asset" not in chunks[0]["text"]


class TestBuildChunksCache:
    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(chunking, "CACHE_DIR", tmp_path)
        cfg = RetrievalConfig(chunk_size=128, chunk_overlap=0)
        first = chunking.build_chunks(cfg, pages([LONG_PAGE, SHORT_PAGE]))
        again = chunking.build_chunks(cfg, None)  # must come from cache, not PAGES_PATH
        assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in again]
        assert all(c["doc_name"] == "DOC_2020_10K" for c in first)
