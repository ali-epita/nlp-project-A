"""Loading and shaping the FinanceBench ground truth.

The open source split has 150 questions. Each question references a single
document and one to three evidence items, each carrying the exact evidence text
and a zero-based page number. A question is answerable by our system only if
its document made it into the extracted corpus, so the loader can restrict the
evaluation set to the documents that were actually ingested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import GOLD_PATH, PAGES_PATH
from .io import read_jsonl


@dataclass
class Question:
    financebench_id: str
    company: str
    doc_name: str
    question_type: str
    question_reasoning: str | None
    question: str
    answer: str
    justification: str | None
    evidence_texts: list[str] = field(default_factory=list)
    evidence_pages: list[int] = field(default_factory=list)

    @classmethod
    def from_record(cls, r: dict) -> "Question":
        return cls(
            financebench_id=r["financebench_id"],
            company=r["company"],
            doc_name=r["doc_name"],
            question_type=r["question_type"],
            question_reasoning=r.get("question_reasoning"),
            question=r["question"],
            answer=r["answer"],
            justification=r.get("justification"),
            evidence_texts=[e["evidence_text"] for e in r["evidence"]],
            evidence_pages=[e["evidence_page_num"] for e in r["evidence"]],
        )


def load_questions(gold_path: Path = GOLD_PATH) -> list[Question]:
    return [Question.from_record(r) for r in read_jsonl(gold_path)]


def load_answerable_questions(
    gold_path: Path = GOLD_PATH, pages_path: Path = PAGES_PATH
) -> tuple[list[Question], list[Question]]:
    """Split questions into (answerable, unanswerable) given the extracted corpus."""
    questions = load_questions(gold_path)
    ingested_docs = {p["doc_name"] for p in read_jsonl(pages_path)}
    answerable = [q for q in questions if q.doc_name in ingested_docs]
    missing = [q for q in questions if q.doc_name not in ingested_docs]
    return answerable, missing
