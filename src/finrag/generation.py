"""Answer generation with local models served by Ollama.

Three context modes support the ablation design:

- retrieved: passages come from a stored retrieval run (the RAG system proper).
- oracle: the gold evidence is passed directly, removing retrieval from the
  equation; the gap between oracle and retrieved accuracy is the retrieval
  bottleneck.
- none: closed book; measures how much the model already knows from
  pre-training, which bounds how much of the RAG score could be memorisation.

Runs are checkpointed after every question batch and resume from the output
file, so an interrupted sweep loses at most a few generations. Every response
records Ollama's token accounting; a prompt that approaches the context window
is flagged so silent truncation (a failure mode observed in the pilot study)
cannot go unnoticed.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

from .config import GENERATIONS_DIR, OLLAMA_URL, GenerationConfig, ensure_dirs
from .io import read_json, write_json
from .metrics import (
    citation_metrics,
    is_abstention,
    numeric_match,
    numeric_match_scale_tolerant,
    token_f1,
)
from .qa import Question

RAG_SYSTEM = (
    "You are a careful financial analyst. You answer questions about SEC filings "
    "strictly from the passages you are given, never from memory."
)

RAG_TEMPLATE = """Answer the question using ONLY the numbered passages below.
Cite the passage number in square brackets, like [2], after every fact you use.
Be concise and state numeric answers with their units.
If the passages do not contain the information needed, reply exactly:
"Insufficient evidence in the provided documents."

Passages:
{passages}

Question: {question}

Answer:"""

CLOSED_BOOK_SYSTEM = "You are a financial analyst with broad knowledge of public company filings."

CLOSED_BOOK_TEMPLATE = """Answer the question from your own knowledge. Be concise and state numeric
answers with their units. Do not say you need documents; give your best answer.

Question: {question}

Answer:"""


def format_passages(passages: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(passages, 1):
        pages = (
            f"page {p['page_start']}"
            if p["page_start"] == p["page_end"]
            else f"pages {p['page_start']}-{p['page_end']}"
        )
        blocks.append(f"[{i}] ({p['doc_name']}, {pages})\n{p['text']}")
    return "\n\n".join(blocks)


def build_prompt(question: str, passages: list[dict], context_mode: str) -> tuple[str, str]:
    """Return (system, user) messages for the given context mode."""
    if context_mode == "none":
        return CLOSED_BOOK_SYSTEM, CLOSED_BOOK_TEMPLATE.format(question=question)
    return RAG_SYSTEM, RAG_TEMPLATE.format(
        passages=format_passages(passages), question=question
    )


def ollama_chat(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    num_ctx: int = 16384,
    max_tokens: int = 512,
    url: str = OLLAMA_URL,
    timeout: int = 600,
) -> dict:
    """One chat completion; returns the text plus token accounting."""
    t0 = time.time()
    resp = requests.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": max_tokens,
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    prompt_tokens = data.get("prompt_eval_count", 0)
    # A prompt at 98 percent of the window has almost certainly been cut.
    truncation_suspected = prompt_tokens >= int(num_ctx * 0.98)
    return {
        "text": data["message"]["content"].strip(),
        "prompt_tokens": prompt_tokens,
        "output_tokens": data.get("eval_count", 0),
        "seconds": time.time() - t0,
        "truncation_suspected": truncation_suspected,
    }


def score_generation(answer: str, passages: list[dict], q: Question) -> dict:
    scores = {
        "numeric_correct": numeric_match(answer, q.answer),
        "numeric_correct_scale_tolerant": numeric_match_scale_tolerant(answer, q.answer),
        "token_f1": token_f1(answer, q.answer),
        "abstained": is_abstention(answer),
    }
    scores.update(citation_metrics(answer, passages, q))
    return scores


def oracle_passages(q: Question) -> list[dict]:
    """The gold evidence formatted as passages."""
    return [
        {
            "doc_name": q.doc_name,
            "page_start": page,
            "page_end": page,
            "text": text,
            "chunk_id": -1,
        }
        for text, page in zip(q.evidence_texts, q.evidence_pages)
    ]


class GenerationRun:
    """A checkpointed, resumable generation run over the evaluation set."""

    def __init__(self, cfg: GenerationConfig, out_dir: Path = GENERATIONS_DIR):
        ensure_dirs()
        self.cfg = cfg
        self.path = out_dir / f"{cfg.run_id()}.json"
        if self.path.exists():
            self.record = read_json(self.path)
            done = len(self.record["generations"])
            if done:
                print(f"Resuming: {done} questions already generated.")
        else:
            self.record = {
                "label": cfg.label,
                "config": cfg.__dict__,
                "generations": [],
            }

    def done_ids(self) -> set[str]:
        return {g["financebench_id"] for g in self.record["generations"]}

    def run(
        self,
        questions: list[Question],
        passages_for: dict[str, list[dict]],
        checkpoint_every: int = 10,
    ) -> dict:
        done = self.done_ids()
        todo = [q for q in questions if q.financebench_id not in done]
        t0 = time.time()
        for i, q in enumerate(todo, 1):
            passages = passages_for.get(q.financebench_id, [])
            if self.cfg.context_mode != "none":
                passages = passages[: self.cfg.k]
            system, user = build_prompt(q.question, passages, self.cfg.context_mode)
            result = ollama_chat(
                self.cfg.gen_model,
                system,
                user,
                temperature=self.cfg.temperature,
                num_ctx=self.cfg.num_ctx,
                max_tokens=self.cfg.max_tokens,
            )
            answer = result.pop("text")
            entry = {
                "financebench_id": q.financebench_id,
                "question": q.question,
                "gold_answer": q.answer,
                "answer": answer,
                "passage_chunk_ids": [p.get("chunk_id", -1) for p in passages],
                "scores": score_generation(answer, passages, q),
                **result,
            }
            self.record["generations"].append(entry)
            if i % checkpoint_every == 0 or i == len(todo):
                self.save()
                elapsed = time.time() - t0
                print(
                    f"[{len(done) + i:3d}/{len(questions)}] "
                    f"{elapsed/60:.1f} min elapsed, {elapsed/i:.1f} s/question"
                )
        self.save()
        return self.record

    def save(self) -> None:
        write_json(self.path, self.record)
