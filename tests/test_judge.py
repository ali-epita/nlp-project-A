"""Unit tests for verdict parsing, prompt building, and agreement statistics."""

import pytest

from finrag.judge import build_judge_prompt, cohen_kappa, parse_verdict


class TestParseVerdict:
    def test_clean_json(self):
        v = parse_verdict(
            '{"correctness": 2, "groundedness": 1, "citations_supported": true, '
            '"hallucination": false, "rationale": "ok"}',
            "test",
        )
        assert v.correctness == 2 and v.groundedness == 1 and v.judge == "test"

    def test_json_wrapped_in_prose(self):
        text = 'Here is my verdict:\n```json\n{"correctness": 0, "groundedness": 0, "citations_supported": false, "hallucination": true, "rationale": "wrong"}\n```'
        v = parse_verdict(text, "test")
        assert v.correctness == 0 and v.hallucination is True

    def test_null_groundedness(self):
        v = parse_verdict('{"correctness": 1, "groundedness": null, "citations_supported": false, "hallucination": false, "rationale": ""}', "t")
        assert v.groundedness is None

    def test_out_of_range_rejected(self):
        assert parse_verdict('{"correctness": 5, "groundedness": 1}', "t") is None

    def test_garbage_rejected(self):
        assert parse_verdict("I think the answer is fine.", "t") is None


class TestPromptBuilding:
    def test_passages_numbered(self):
        p = build_judge_prompt("Q?", "ref", "ans", ["first passage", "second passage"])
        assert "[1] first passage" in p and "[2] second passage" in p

    def test_closed_book_variant(self):
        p = build_judge_prompt("Q?", "ref", "ans", [])
        assert "closed book" in p and '"groundedness": null' in p


class TestCohenKappa:
    def test_perfect_agreement(self):
        assert cohen_kappa([0, 1, 2, 1], [0, 1, 2, 1]) == pytest.approx(1.0)

    def test_no_better_than_chance(self):
        a = [0, 0, 1, 1]
        b = [0, 1, 0, 1]
        assert abs(cohen_kappa(a, b)) < 0.5

    def test_weighted_forgives_near_misses(self):
        a = [0, 1, 2, 2, 1, 0]
        b = [0, 2, 1, 2, 1, 1]
        assert cohen_kappa(a, b, weighted=True) > cohen_kappa(a, b)
