"""Automated retrieval of the FinanceBench source PDFs.

Primary source is the doc_link URL from the document catalog. A substantial
fraction of those links has rotted (the pilot study measured 19 dead links out
of 84), so every failed download falls back to the official FinanceBench GitHub
repository, which mirrors the PDFs under pdfs/<doc_name>.pdf. Each document's
outcome is recorded in a manifest so the report can state exactly where every
file came from.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from ..config import DOC_INFO_PATH, DOWNLOAD_MANIFEST_PATH, GOLD_PATH, PDF_DIR, ensure_dirs
from ..io import read_jsonl, write_json

GITHUB_PDF_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{doc_name}.pdf"

# SEC EDGAR blocks anonymous scripted clients; they ask for a descriptive
# User-Agent carrying a contact address.
HEADERS = {"User-Agent": "FinanceBench-RAG-course-project/0.1 (ali.h.cherri@gmail.com)"}

TIMEOUT = 30
RETRIES = 3
MIN_PDF_BYTES = 10_000


def _fetch(url: str, session: requests.Session) -> bytes | None:
    """Download url with retries; return bytes only if they look like a PDF."""
    for attempt in range(RETRIES):
        try:
            resp = session.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.content
            if data[:5] == b"%PDF-" and len(data) >= MIN_PDF_BYTES:
                return data
            return None  # HTML error page or truncated body; retrying will not help
        except requests.RequestException:
            if attempt < RETRIES - 1:
                time.sleep(2**attempt)
    return None


def _download_one(doc: dict, session: requests.Session) -> dict:
    doc_name = doc["doc_name"]
    target = PDF_DIR / f"{doc_name}.pdf"
    if target.exists() and target.stat().st_size >= MIN_PDF_BYTES:
        data = target.read_bytes()
        if data[:5] == b"%PDF-":
            return {
                "doc_name": doc_name,
                "source": "cached",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }

    # Politeness delay for SEC servers, which rate limit aggressively.
    if "sec.gov" in (doc.get("doc_link") or ""):
        time.sleep(0.3)

    for source, url in (
        ("doc_link", doc.get("doc_link")),
        ("github", GITHUB_PDF_URL.format(doc_name=doc_name)),
    ):
        if not url:
            continue
        data = _fetch(url, session)
        if data is not None:
            target.write_bytes(data)
            return {
                "doc_name": doc_name,
                "source": source,
                "url": url,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    return {"doc_name": doc_name, "source": "failed"}


def download_corpus(only_needed: bool = True, max_workers: int = 6) -> dict:
    """Download the PDFs and write a manifest; returns the manifest dict."""
    ensure_dirs()
    docs = read_jsonl(DOC_INFO_PATH)
    if only_needed:
        needed = {g["doc_name"] for g in read_jsonl(GOLD_PATH)}
        docs = [d for d in docs if d["doc_name"] in needed]

    results: list[dict] = []
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_download_one, d, session): d for d in docs}
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                status = r["source"]
                print(f"[{i:3d}/{len(docs)}] {r['doc_name']:45s} {status}")

    results.sort(key=lambda r: r["doc_name"])
    ok = [r for r in results if r["source"] != "failed"]
    manifest = {
        "n_requested": len(docs),
        "n_downloaded": len(ok),
        "n_failed": len(docs) - len(ok),
        "by_source": {
            s: sum(1 for r in results if r["source"] == s)
            for s in ("cached", "doc_link", "github", "failed")
        },
        "documents": results,
    }
    write_json(DOWNLOAD_MANIFEST_PATH, manifest)
    print(
        f"\n{manifest['n_downloaded']}/{manifest['n_requested']} PDFs available "
        f"({manifest['by_source']})"
    )
    return manifest
