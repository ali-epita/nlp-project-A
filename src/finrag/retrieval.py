"""Retrieval strategies over the chunked corpus.

Four strategies share one interface:

- dense: cosine similarity over normalised embeddings (FAISS inner product).
- bm25: lexical Okapi BM25 baseline.
- hybrid: convex combination of min-max normalised dense and BM25 scores.
- rerank: dense candidate pool rescored by a cross-encoder.

An optional metadata filter restricts candidates to documents whose company
(and, when stated, fiscal period) is mentioned in the question text. The filter
reads nothing from the gold record beyond the question string, so it is
information a deployed system would also have.
"""

from __future__ import annotations

import functools
import re

import numpy as np

from .config import DOC_INFO_PATH, RetrievalConfig
from .embeddings import embed_texts, get_model
from .io import read_jsonl

_WORD_RE = re.compile(r"[a-z0-9]+")

# Common shorthand for companies whose full catalog name rarely appears verbatim.
COMPANY_ALIASES = {
    "amex": "American Express",
    "amcor": "Amcor",
    "jpmorgan chase": "JPMorgan",
    "jp morgan": "JPMorgan",
    "coca cola": "Coca-Cola",
    "coca-cola": "Coca-Cola",
    "j&j": "Johnson & Johnson",
    "pepsi": "PepsiCo",
}


def _squash(text: str) -> str:
    return "".join(_WORD_RE.findall(text.lower()))


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@functools.lru_cache(maxsize=1)
def _company_catalog() -> list[tuple[str, str, frozenset[tuple[str, int]]]]:
    """(company, squashed_name, {(doc_name, period)}) for every catalog company."""
    docs = read_jsonl(DOC_INFO_PATH)
    by_company: dict[str, set[tuple[str, int]]] = {}
    for d in docs:
        by_company.setdefault(d["company"], set()).add((d["doc_name"], d["doc_period"]))
    return [(c, _squash(c), frozenset(v)) for c, v in by_company.items()]


def match_question_metadata(question: str) -> tuple[set[str] | None, set[int]]:
    """Return (allowed doc names or None, fiscal years mentioned) for a question."""
    squashed = _squash(question)
    lowered = question.lower()
    matched: set[tuple[str, int]] = set()
    for company, sq_name, docs in _company_catalog():
        if sq_name and sq_name in squashed:
            matched |= docs
    if not matched:
        for alias, company in COMPANY_ALIASES.items():
            if alias in lowered:
                for c, _, docs in _company_catalog():
                    if c == company:
                        matched |= docs
    years = {int(y) for y in re.findall(r"(?:FY\s?)?((?:19|20)\d{2})", question)}
    if not matched:
        return None, years
    if years:
        year_docs = {name for name, period in matched if period in years}
        if year_docs:
            return year_docs, years
    return {name for name, _ in matched}, years


class Retriever:
    """One retrieval configuration, ready to answer queries."""

    def __init__(
        self,
        cfg: RetrievalConfig,
        chunks: list[dict],
        matrix: np.ndarray | None,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.chunks = chunks
        self.matrix = matrix
        self.device = device
        self._faiss_index = None
        self._bm25 = None
        self._reranker = None

    # Lazy construction keeps cheap strategies cheap.
    @property
    def faiss_index(self):
        if self._faiss_index is None:
            import faiss

            index = faiss.IndexFlatIP(self.matrix.shape[1])
            index.add(self.matrix)
            self._faiss_index = index
        return self._faiss_index

    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])
        return self._bm25

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.cfg.reranker_model, device=self.device)
        return self._reranker

    def _allowed_ids(self, question: str) -> np.ndarray | None:
        if not self.cfg.metadata_filter:
            return None
        allowed_docs, _ = match_question_metadata(question)
        if allowed_docs is None:
            return None
        ids = np.array(
            [c["chunk_id"] for c in self.chunks if c["doc_name"] in allowed_docs], dtype=np.int64
        )
        return ids if len(ids) else None

    def _dense_scores(self, query_vec: np.ndarray) -> np.ndarray:
        return self.matrix @ query_vec

    def _bm25_scores(self, question: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(_tokenize(question)), dtype=np.float32)

    @staticmethod
    def _top_ids(scores: np.ndarray, k: int, allowed: np.ndarray | None) -> np.ndarray:
        if allowed is not None:
            mask = np.full(scores.shape, -np.inf, dtype=np.float32)
            mask[allowed] = scores[allowed]
            scores = mask
        k = min(k, (scores > -np.inf).sum())
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])]

    @staticmethod
    def _minmax(scores: np.ndarray) -> np.ndarray:
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-9:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def query_vector(self, question: str) -> np.ndarray:
        return embed_texts([question], self.cfg.embedding_model, self.device, is_query=True)[0]

    def search(self, question: str, query_vec: np.ndarray | None = None) -> list[dict]:
        """Return the top-k chunks for a question, each with a retrieval score."""
        cfg = self.cfg
        allowed = self._allowed_ids(question)
        needs_dense = cfg.strategy in ("dense", "hybrid", "rerank")
        if needs_dense and query_vec is None:
            query_vec = self.query_vector(question)

        if cfg.strategy == "dense":
            if allowed is None:
                scores, ids = self.faiss_index.search(query_vec[None, :], cfg.k)
                ranked = list(zip(ids[0].tolist(), scores[0].tolist()))
            else:
                s = self._dense_scores(query_vec)
                top = self._top_ids(s, cfg.k, allowed)
                ranked = [(int(i), float(s[i])) for i in top]

        elif cfg.strategy == "bm25":
            s = self._bm25_scores(question)
            top = self._top_ids(s, cfg.k, allowed)
            ranked = [(int(i), float(s[i])) for i in top]

        elif cfg.strategy == "hybrid":
            dense = self._minmax(self._dense_scores(query_vec))
            lexical = self._minmax(self._bm25_scores(question))
            s = cfg.alpha * dense + (1 - cfg.alpha) * lexical
            top = self._top_ids(s, cfg.k, allowed)
            ranked = [(int(i), float(s[i])) for i in top]

        elif cfg.strategy == "rerank":
            s = self._dense_scores(query_vec)
            pool = self._top_ids(s, cfg.rerank_pool, allowed)
            pairs = [(question, self.chunks[i]["text"]) for i in pool]
            ce_scores = self.reranker.predict(pairs, show_progress_bar=False)
            order = np.argsort(-np.asarray(ce_scores))[: cfg.k]
            ranked = [(int(pool[j]), float(ce_scores[j])) for j in order]

        else:
            raise ValueError(f"Unknown strategy: {cfg.strategy}")

        return [
            {**self.chunks[i], "score": score, "rank": rank}
            for rank, (i, score) in enumerate(ranked, 1)
        ]
