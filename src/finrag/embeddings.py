"""Dense embeddings with per-model query and passage prefixes and disk caching.

Embedding matrices are expensive to build (minutes per configuration on a GPU,
hours on CPU), so each one is cached under a key derived from the chunk set and
the model name. Asymmetric models (E5, BGE) need their documented prefixes;
omitting them costs several recall points, which would contaminate the model
comparison.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import CACHE_DIR, RetrievalConfig, ensure_dirs

# (query_prefix, passage_prefix) as documented by each model's authors.
MODEL_PREFIXES: dict[str, tuple[str, str]] = {
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-large-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
    "intfloat/e5-large-v2": ("query: ", "passage: "),
    "thenlper/gte-base": ("", ""),
    "sentence-transformers/all-MiniLM-L6-v2": ("", ""),
}

_MODEL_CACHE: dict[tuple[str, str], object] = {}


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    if device == "cuda":
        import torch

        return f"cuda ({torch.cuda.get_device_name(0)})"
    return device


def get_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    key = (model_name, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[key]


def embed_texts(
    texts: list[str], model_name: str, device: str, is_query: bool = False, batch_size: int = 64
) -> np.ndarray:
    prefix = MODEL_PREFIXES.get(model_name, ("", ""))[0 if is_query else 1]
    model = get_model(model_name, device)
    matrix = model.encode(
        [prefix + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 1000,
        convert_to_numpy=True,
    )
    return matrix.astype(np.float32)


def load_or_build_embeddings(
    cfg: RetrievalConfig, chunks: list[dict], device: str
) -> np.ndarray:
    """Return the (n_chunks, dim) matrix for cfg, building and caching if needed."""
    ensure_dirs()
    cache_path = CACHE_DIR / f"emb_{cfg.embed_key()}.npy"
    if cache_path.exists():
        matrix = np.load(cache_path)
        if matrix.shape[0] == len(chunks):
            return matrix

    t0 = time.time()
    print(f"Embedding {len(chunks):,} chunks with {cfg.embedding_model} on {device}")
    matrix = embed_texts([c["text"] for c in chunks], cfg.embedding_model, device)
    np.save(cache_path, matrix)
    rate = len(chunks) / max(time.time() - t0, 1e-9)
    print(f"  {matrix.shape[0]:,} x {matrix.shape[1]} in {time.time()-t0:.0f}s ({rate:.0f} chunks/s)")
    return matrix
