"""Tests for retrieval strategies and metadata filtering on a synthetic corpus."""

import numpy as np
import pytest

from finrag.config import RetrievalConfig
from finrag.retrieval import Retriever, match_question_metadata


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


CHUNKS = [
    {"chunk_id": 0, "doc_name": "3M_2018_10K", "page_start": 0, "page_end": 0,
     "text": "capital expenditure purchases of property plant and equipment"},
    {"chunk_id": 1, "doc_name": "3M_2018_10K", "page_start": 1, "page_end": 1,
     "text": "legal proceedings and risk factors discussion"},
    {"chunk_id": 2, "doc_name": "APPLE_2020_10K", "page_start": 0, "page_end": 0,
     "text": "iphone revenue and services segment growth"},
]

MATRIX = np.stack([unit([1, 0, 0]), unit([0, 1, 0]), unit([0.9, 0.1, 0.4])])


class TestMetadataMatching:
    def test_company_and_year(self):
        docs, years = match_question_metadata("What was 3M's FY2018 capital expenditure?")
        assert docs is not None and any("3M_2018" in d for d in docs)
        assert 2018 in years

    def test_no_company_mentioned(self):
        docs, years = match_question_metadata("What was the capital expenditure that year?")
        assert docs is None

    def test_year_narrows_documents(self):
        docs_2018, _ = match_question_metadata("3M capex in FY2018")
        docs_all, _ = match_question_metadata("3M capex history")
        assert docs_2018 is not None and docs_all is not None
        assert len(docs_2018) <= len(docs_all)


class TestStrategies:
    def make(self, **overrides):
        cfg = RetrievalConfig(**{"strategy": "dense", "k": 2, **overrides})
        return Retriever(cfg, CHUNKS, MATRIX.copy(), device="cpu")

    def test_dense_ranks_by_cosine(self):
        r = self.make()
        out = r.search("q", query_vec=unit([1, 0, 0]))
        assert [c["chunk_id"] for c in out] == [0, 2]
        assert out[0]["score"] >= out[1]["score"]

    def test_bm25_prefers_exact_terms(self):
        r = self.make(strategy="bm25")
        out = r.search("iphone services revenue")
        assert out[0]["chunk_id"] == 2

    def test_hybrid_combines_signals(self):
        r = self.make(strategy="hybrid", alpha=0.5)
        out = r.search("capital expenditure", query_vec=unit([1, 0, 0]))
        assert out[0]["chunk_id"] == 0

    def test_metadata_filter_excludes_other_companies(self):
        r = self.make(metadata_filter=True)
        out = r.search("Apple iphone revenue in 2020", query_vec=unit([1, 0, 0]))
        assert all(c["doc_name"].startswith("APPLE") for c in out)

    def test_rank_field_is_one_based(self):
        out = self.make().search("q", query_vec=unit([0, 1, 0]))
        assert [c["rank"] for c in out] == [1, 2]
