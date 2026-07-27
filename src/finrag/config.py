"""Repository paths and experiment configuration.

Every experiment is described by a frozen dataclass whose content hash acts as
the run identifier. Runs, chunk sets, and embedding matrices are cached under
these identifiers so any sweep can be interrupted and resumed without redoing
finished work.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PDF_DIR = DATA / "pdfs"
CORPUS_DIR = DATA / "corpus"
CACHE_DIR = DATA / "cache"
RESULTS = ROOT / "results"
RUNS_DIR = RESULTS / "runs"
GENERATIONS_DIR = RESULTS / "generations"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

DOC_INFO_PATH = DATA / "financebench_document_information.jsonl"
GOLD_PATH = DATA / "financebench_open_source.jsonl"
PAGES_PATH = CORPUS_DIR / "pages.jsonl"
DOWNLOAD_MANIFEST_PATH = PDF_DIR / "manifest.json"

# Reference tokenizer used for chunk boundaries. Keeping boundaries fixed while
# swapping embedding models makes the embedding comparison a controlled one:
# every model sees exactly the same chunks.
CHUNK_TOKENIZER = "bert-base-uncased"

OLLAMA_URL = "http://127.0.0.1:11434"


def _short_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


@dataclass(frozen=True)
class RetrievalConfig:
    """One point in the retrieval design space."""

    label: str = "baseline"
    strategy: str = "dense"  # dense | bm25 | hybrid | rerank
    chunker: str = "token"  # token | page | structural
    chunk_size: int = 512  # tokens per chunk
    chunk_overlap: int = 64  # tokens shared between consecutive chunks
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    k: int = 5  # passages returned to the generator
    alpha: float = 0.5  # hybrid only: weight of the dense score
    rerank_pool: int = 50  # rerank only: candidates given to the cross-encoder
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    metadata_filter: bool = False  # restrict candidates to docs matching the question

    def chunk_key(self) -> str:
        """Cache key for the chunk set (independent of the embedding model)."""
        return _short_hash(
            {
                "chunker": self.chunker,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "tokenizer": CHUNK_TOKENIZER,
            }
        )

    def embed_key(self) -> str:
        """Cache key for the embedding matrix of this chunk set and model."""
        return _short_hash({"chunks": self.chunk_key(), "model": self.embedding_model})

    def run_id(self) -> str:
        """Identifier for the full retrieval run (excludes the display label)."""
        payload = asdict(self)
        payload.pop("label")
        return _short_hash(payload)


@dataclass(frozen=True)
class GenerationConfig:
    """One generation run: a model applied to some source of context."""

    label: str = "gen"
    gen_model: str = "qwen2.5:7b-instruct"
    context_mode: str = "retrieved"  # retrieved | oracle | none
    retrieval_run_id: str = ""  # which retrieval run supplies the passages
    k: int = 10  # passages actually placed in the prompt
    temperature: float = 0.0
    num_ctx: int = 16384  # context window; validated against measured prompt sizes
    max_tokens: int = 512

    def run_id(self) -> str:
        payload = asdict(self)
        payload.pop("label")
        return _short_hash(payload)


def ensure_dirs() -> None:
    for d in (PDF_DIR, CORPUS_DIR, CACHE_DIR, RUNS_DIR, GENERATIONS_DIR, FIGURES, TABLES):
        d.mkdir(parents=True, exist_ok=True)
