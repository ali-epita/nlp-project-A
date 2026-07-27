"""Unit tests for the deterministic metrics."""

import pytest

from finrag.metrics import (
    citation_metrics,
    evidence_coverage,
    is_abstention,
    numeric_match,
    numeric_match_scale_tolerant,
    parse_citations,
    parse_numbers,
    retrieval_metrics,
    strict_hit,
    token_f1,
)
from finrag.qa import Question


def make_question(**overrides) -> Question:
    base = dict(
        financebench_id="q1",
        company="TestCo",
        doc_name="TESTCO_2020_10K",
        question_type="metrics-generated",
        question_reasoning="Information extraction",
        question="What was TestCo's FY2020 revenue?",
        answer="$1,577.00",
        justification=None,
        evidence_texts=["Total revenue for fiscal 2020 was 1,577 million dollars as reported"],
        evidence_pages=[41],
    )
    base.update(overrides)
    return Question(**base)


def chunk(doc="TESTCO_2020_10K", start=41, end=41, text="irrelevant filler text", cid=0):
    return {"doc_name": doc, "page_start": start, "page_end": end, "text": text, "chunk_id": cid}


class TestParseNumbers:
    def test_currency_with_thousands_separator(self):
        assert parse_numbers("$1,577.00") == [1577.0]

    def test_percentage(self):
        assert parse_numbers("margin fell to 8.7%") == [8.7]

    def test_scale_words(self):
        assert parse_numbers("USD 1.5 billion") == [1.5e9]
        assert parse_numbers("1,022 million") == [1.022e9]

    def test_parenthesised_negative(self):
        assert parse_numbers("(3.2)") == [-3.2]

    def test_no_numbers(self):
        assert parse_numbers("no figures here") == []


class TestNumericMatch:
    def test_exact_match(self):
        assert numeric_match("Revenue was $1,577.00 in FY2020", "$1577.00") is True

    def test_within_tolerance(self):
        assert numeric_match("about $1,580", "$1577.00", rel_tol=0.01) is True

    def test_wrong_value(self):
        assert numeric_match("Revenue was $2,000", "$1577.00") is False

    def test_non_numeric_gold_excluded(self):
        assert numeric_match("anything", "Yes, margins improved.") is None

    def test_scale_tolerant_rescues_unit_mismatch(self):
        pred = "Capital expenditure was 1,577 million dollars"
        assert numeric_match(pred, "$1577.00") is False  # 1.577e9 vs 1577
        assert numeric_match_scale_tolerant(pred, "$1577.00") is True

    def test_scale_tolerant_percent_fraction(self):
        assert numeric_match_scale_tolerant("the ratio was 0.087", "8.7%") is True


class TestTokenF1:
    def test_identical(self):
        assert token_f1("the revenue rose", "The revenue rose.") == pytest.approx(1.0)

    def test_disjoint(self):
        assert token_f1("alpha beta", "gamma delta") == 0.0


class TestRetrievalMetrics:
    def test_perfect_first_rank(self):
        q = make_question()
        retrieved = [chunk(text=q.evidence_texts[0]), chunk(start=2, end=2, cid=1)]
        m = retrieval_metrics(retrieved, q)
        assert m["recall"] == 1.0
        assert m["mrr"] == 1.0
        assert m["ndcg"] == pytest.approx(1.0)
        assert m["doc_accuracy"] == 1.0
        assert m["evidence_coverage"] > 0.9

    def test_miss_everything(self):
        q = make_question()
        retrieved = [chunk(doc="OTHER_2019_10K", start=3, end=3)]
        m = retrieval_metrics(retrieved, q)
        assert m["recall"] == 0.0
        assert m["doc_accuracy"] == 0.0
        assert m["mrr"] == 0.0

    def test_right_doc_wrong_page(self):
        q = make_question()
        retrieved = [chunk(start=7, end=7)]
        m = retrieval_metrics(retrieved, q)
        assert m["recall"] == 0.0
        assert m["doc_accuracy"] == 1.0

    def test_page_span_covers_gold(self):
        q = make_question()
        retrieved = [chunk(start=40, end=42)]
        assert retrieval_metrics(retrieved, q)["recall"] == 1.0

    def test_second_rank_mrr(self):
        q = make_question()
        retrieved = [chunk(start=7, end=7), chunk(start=41, end=41, cid=1)]
        assert retrieval_metrics(retrieved, q)["mrr"] == pytest.approx(0.5)


class TestEvidenceCoverage:
    def test_boundary_loss_detected(self):
        q = make_question()
        # Gold page retrieved, but the chunk holds none of the evidence text.
        retrieved = [chunk(text="completely different content about something else entirely here")]
        assert evidence_coverage(retrieved, q) < 0.1
        assert strict_hit(retrieved, q) is False

    def test_full_evidence_in_one_chunk(self):
        q = make_question()
        retrieved = [chunk(text="Preamble. " + q.evidence_texts[0] + " Postamble.")]
        assert evidence_coverage(retrieved, q) == pytest.approx(1.0)
        assert strict_hit(retrieved, q) is True


class TestCitations:
    def test_parse_and_bounds(self):
        assert parse_citations("see [1] and [3], also [12]", n_passages=2) == [0]

    def test_valid_citation_points_at_gold_page(self):
        q = make_question()
        passages = [chunk(start=7, end=7), chunk(start=41, end=41, cid=1)]
        m = citation_metrics("Revenue was $1,577 [2]", passages, q)
        assert m["has_citation"] and m["citation_valid"]

    def test_invalid_citation(self):
        q = make_question()
        passages = [chunk(start=7, end=7)]
        m = citation_metrics("Revenue was $1,577 [1]", passages, q)
        assert m["has_citation"] and not m["citation_valid"]


def test_abstention_marker():
    assert is_abstention("Insufficient evidence in the provided documents.")
    assert not is_abstention("The revenue was $5")
