"""Tests for config hashing and ground truth loading."""

from finrag.config import GenerationConfig, RetrievalConfig
from finrag.qa import load_questions


class TestConfigHashing:
    def test_run_id_stable_and_label_free(self):
        a = RetrievalConfig(label="one", k=5)
        b = RetrievalConfig(label="two", k=5)
        assert a.run_id() == b.run_id()

    def test_run_id_changes_with_content(self):
        assert RetrievalConfig(k=5).run_id() != RetrievalConfig(k=10).run_id()

    def test_chunk_key_shared_across_embedding_models(self):
        a = RetrievalConfig(embedding_model="BAAI/bge-base-en-v1.5")
        b = RetrievalConfig(embedding_model="intfloat/e5-base-v2")
        assert a.chunk_key() == b.chunk_key()
        assert a.embed_key() != b.embed_key()

    def test_generation_run_id(self):
        a = GenerationConfig(gen_model="m1")
        b = GenerationConfig(gen_model="m2")
        assert a.run_id() != b.run_id()


class TestGroundTruth:
    def test_full_load(self):
        questions = load_questions()
        assert len(questions) == 150
        assert all(q.question and q.answer and q.doc_name for q in questions)

    def test_evidence_alignment(self):
        questions = load_questions()
        for q in questions:
            assert len(q.evidence_texts) == len(q.evidence_pages)
            assert all(isinstance(p, int) and p >= 0 for p in q.evidence_pages)

    def test_question_types(self):
        types = {q.question_type for q in load_questions()}
        assert types == {"metrics-generated", "domain-relevant", "novel-generated"}
