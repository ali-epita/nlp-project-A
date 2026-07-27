"""Tests for prompt construction and generation scoring (no server required)."""

from finrag.generation import build_prompt, format_passages, oracle_passages, score_generation
from finrag.metrics import ABSTAIN_MARKER

from .test_metrics import chunk, make_question


class TestPromptConstruction:
    def test_passages_numbered_with_provenance(self):
        passages = [chunk(text="alpha"), chunk(start=2, end=3, text="beta", cid=1)]
        block = format_passages(passages)
        assert "[1] (TESTCO_2020_10K, page 41)" in block
        assert "[2] (TESTCO_2020_10K, pages 2-3)" in block

    def test_rag_prompt_carries_instructions(self):
        _, user = build_prompt("What was revenue?", [chunk()], "retrieved")
        assert "ONLY" in user and ABSTAIN_MARKER.lower() in user.lower()

    def test_closed_book_prompt_has_no_passages(self):
        system, user = build_prompt("What was revenue?", [], "none")
        assert "passage" not in user.lower()


class TestScoring:
    def test_correct_cited_answer(self):
        q = make_question()
        passages = [chunk(text=q.evidence_texts[0])]
        s = score_generation("Revenue was $1,577.00 [1]", passages, q)
        assert s["numeric_correct"] is True
        assert s["has_citation"] and s["citation_valid"]
        assert not s["abstained"]

    def test_abstention_scored(self):
        q = make_question()
        s = score_generation("Insufficient evidence in the provided documents.", [], q)
        assert s["abstained"] and s["numeric_correct"] is False


def test_oracle_passages_mirror_evidence():
    q = make_question()
    ps = oracle_passages(q)
    assert len(ps) == 1
    assert ps[0]["text"] == q.evidence_texts[0]
    assert ps[0]["page_start"] == q.evidence_pages[0]
