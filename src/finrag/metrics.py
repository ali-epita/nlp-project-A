"""Deterministic evaluation metrics.

Retrieval is scored at page level against FinanceBench's page-annotated
evidence, with an n-gram containment measure catching the case page-level
metrics hide: the right page retrieved but the evidence text cut away by a
chunk boundary. Generation is scored with a scale-tolerant numeric accuracy
(most FinanceBench answers are dollar amounts, percentages, or ratios), a
SQuAD-style token F1, and deterministic citation checks.
"""

from __future__ import annotations

import math
import re
import string

from .qa import Question

# ---------------------------------------------------------------------------
# Text normalisation and token F1
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    text = text.lower().translate(_PUNCT_TABLE)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common: dict[str, int] = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Numeric accuracy
# ---------------------------------------------------------------------------

_SCALE_WORDS = {
    "thousand": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "trillion": 1e12,
}

_NUMBER_RE = re.compile(
    r"\(?-?\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\)?\s*(%|thousand|million|mn|billion|bn|trillion)?",
    re.IGNORECASE,
)


def parse_numbers(text: str) -> list[float]:
    """Extract numeric magnitudes; handles $, commas, %, scale words, (negatives)."""
    values: list[float] = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(1).replace(",", "") + (m.group(2) or "")
        value = float(raw)
        suffix = (m.group(3) or "").lower()
        if suffix in _SCALE_WORDS:
            value *= _SCALE_WORDS[suffix]
        whole = m.group(0)
        if whole.strip().startswith("(") and whole.strip().endswith(")"):
            value = -value  # accounting convention for negatives
        elif "-" in whole.split("$")[0]:
            value = -value
        values.append(value)
    return values


# Ratios treated as equivalent under the scale-tolerant match: unit changes
# (thousands, millions, billions) and the percent to fraction conversion.
_SCALE_FACTORS = [1.0, 1e2, 1e3, 1e6, 1e9, 1e-2, 1e-3, 1e-6, 1e-9]


def _close(a: float, b: float, rel_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)


def numeric_match(prediction: str, gold: str, rel_tol: float = 0.01) -> bool | None:
    """Strict match: every gold number appears in the prediction (within rel_tol)."""
    gold_nums = parse_numbers(gold)
    if not gold_nums:
        return None  # question has a non-numeric answer; metric does not apply
    pred_nums = parse_numbers(prediction)
    return all(any(_close(p, g, rel_tol) for p in pred_nums) for g in gold_nums)


def numeric_match_scale_tolerant(prediction: str, gold: str, rel_tol: float = 0.01) -> bool | None:
    gold_nums = parse_numbers(gold)
    if not gold_nums:
        return None
    pred_nums = parse_numbers(prediction)
    return all(
        any(
            _close(abs(p) * f, abs(g), rel_tol)
            for p in pred_nums
            for f in _SCALE_FACTORS
        )
        for g in gold_nums
    )


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def chunk_is_relevant(chunk: dict, q: Question) -> bool:
    """A chunk is relevant if it comes from the gold document and covers a gold page."""
    return chunk["doc_name"] == q.doc_name and any(
        chunk["page_start"] <= p <= chunk["page_end"] for p in q.evidence_pages
    )


_WS_RE = re.compile(r"\s+")


def _ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    tokens = _WS_RE.sub(" ", text.lower()).split()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def evidence_coverage(retrieved: list[dict], q: Question, n: int = 5) -> float:
    """Mean fraction of each evidence item's n-grams present in retrieved chunks."""
    union: set[tuple[str, ...]] = set()
    for c in retrieved:
        union |= _ngrams(c["text"], n)
    fractions = []
    for ev in q.evidence_texts:
        grams = _ngrams(ev, n)
        if not grams:  # evidence shorter than n tokens
            fractions.append(0.0)
            continue
        fractions.append(len(grams & union) / len(grams))
    return sum(fractions) / len(fractions) if fractions else 0.0


def strict_hit(retrieved: list[dict], q: Question, n: int = 5, threshold: float = 0.5) -> bool:
    """True if a single chunk contains at least threshold of some evidence item."""
    for ev in q.evidence_texts:
        grams = _ngrams(ev, n)
        if not grams:
            continue
        for c in retrieved:
            chunk_grams = _ngrams(c["text"], n)
            if len(grams & chunk_grams) / len(grams) >= threshold:
                return True
    return False


def retrieval_metrics(retrieved: list[dict], q: Question) -> dict:
    """All retrieval metrics for one question's ranked list."""
    ranks = [i for i, c in enumerate(retrieved, 1) if chunk_is_relevant(c, q)]
    k = len(retrieved)
    hit = bool(ranks)
    mrr = 1.0 / ranks[0] if ranks else 0.0
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks)
    ideal_r = min(len(set(q.evidence_pages)), k) or 1
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_r + 1))
    return {
        "recall": float(hit),
        "strict_recall": float(strict_hit(retrieved, q)),
        "doc_accuracy": float(any(c["doc_name"] == q.doc_name for c in retrieved)),
        "mrr": mrr,
        "ndcg": dcg / idcg,
        "evidence_coverage": evidence_coverage(retrieved, q),
    }


# ---------------------------------------------------------------------------
# Citation and abstention checks on generated answers
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
ABSTAIN_MARKER = "insufficient evidence"


def parse_citations(answer: str, n_passages: int) -> list[int]:
    """Zero-based indices of validly numbered citations in the answer."""
    cited = {int(m) - 1 for m in _CITATION_RE.findall(answer)}
    return sorted(i for i in cited if 0 <= i < n_passages)


def citation_metrics(answer: str, passages: list[dict], q: Question) -> dict:
    cited = parse_citations(answer, len(passages))
    valid = [i for i in cited if chunk_is_relevant(passages[i], q)]
    return {
        "has_citation": bool(cited),
        "citation_valid": bool(valid),
        "n_citations": len(cited),
    }


def is_abstention(answer: str) -> bool:
    return ABSTAIN_MARKER in answer.lower()
