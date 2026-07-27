"""LLM-as-judge with interchangeable backends.

The primary judge is a local model served by Ollama (free and reproducible, in
the spirit of the project constraint). A second backend shells out to the Codex
CLI (gpt-5.5) and is used only as a cross-check: agreement between the two
judges, reported as Cohen's kappa, stands in for human validation of the judge.

Every backend receives the same prompt and must return the same JSON verdict:

    {"correctness": 0|1|2, "groundedness": 0|1|2 or null,
     "citations_supported": bool, "hallucination": bool, "rationale": str}
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass

import requests

from .config import OLLAMA_URL

JUDGE_SYSTEM = (
    "You are a strict evaluator of answers produced by a question answering "
    "system over financial filings. Judge only against the reference answer "
    "and the provided passages. Respond with a single JSON object and nothing else."
)

JUDGE_TEMPLATE = """Question:
{question}

Reference answer:
{reference}

Reference justification:
{justification}

System answer to evaluate:
{answer}

{passages_block}Grade the system answer:
- correctness: 2 if it is equivalent to the reference answer (numeric values within about 1 percent, wording may differ), 1 if partially correct or correct but incomplete, 0 if incorrect or an unwarranted refusal.
- groundedness: {groundedness_rubric}
- citations_supported: true if the answer's cited passages actually contain the claims they are attached to, false otherwise.
- hallucination: true if the answer states material facts that appear in neither the passages nor the reference answer.

Respond with only this JSON object:
{{"correctness": 0 or 1 or 2, "groundedness": {groundedness_type}, "citations_supported": true or false, "hallucination": true or false, "rationale": "one or two sentences"}}"""

GROUNDEDNESS_RUBRIC = (
    "2 if every claim is supported by the passages, 1 if partially supported, "
    "0 if unsupported or contradicting the passages."
)


@dataclass
class Verdict:
    correctness: int
    groundedness: int | None
    citations_supported: bool
    hallucination: bool
    rationale: str
    judge: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def build_judge_prompt(
    question: str,
    reference: str,
    answer: str,
    passages: list[str],
    justification: str | None = None,
    max_passage_chars: int = 1500,
) -> str:
    if passages:
        listed = "\n\n".join(
            f"[{i}] {p[:max_passage_chars]}" for i, p in enumerate(passages, 1)
        )
        passages_block = f"Passages given to the system:\n{listed}\n\n"
        rubric, gtype = GROUNDEDNESS_RUBRIC, "0 or 1 or 2"
    else:
        passages_block = "The system received no passages (closed book).\n\n"
        rubric, gtype = "null (no passages were provided).", "null"
    return JUDGE_TEMPLATE.format(
        question=question,
        reference=reference,
        justification=justification or "not provided",
        answer=answer or "(empty answer)",
        passages_block=passages_block,
        groundedness_rubric=rubric,
        groundedness_type=gtype,
    )


def parse_verdict(text: str, judge_name: str) -> Verdict | None:
    """Extract and validate the verdict JSON from a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    try:
        correctness = int(obj["correctness"])
        groundedness = obj.get("groundedness")
        groundedness = None if groundedness is None else int(groundedness)
    except (KeyError, TypeError, ValueError):
        return None
    if correctness not in (0, 1, 2) or groundedness not in (None, 0, 1, 2):
        return None
    return Verdict(
        correctness=correctness,
        groundedness=groundedness,
        citations_supported=bool(obj.get("citations_supported", False)),
        hallucination=bool(obj.get("hallucination", False)),
        rationale=str(obj.get("rationale", ""))[:500],
        judge=judge_name,
    )


class OllamaJudge:
    """Primary judge: a local instruction model served by Ollama."""

    def __init__(self, model: str = "qwen2.5:14b-instruct-q4_K_M", url: str = OLLAMA_URL):
        self.model = model
        self.url = url
        self.name = f"ollama:{model}"

    def judge(self, prompt: str, retries: int = 2) -> Verdict | None:
        for _ in range(retries + 1):
            resp = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.0, "num_ctx": 16384},
                },
                timeout=300,
            )
            resp.raise_for_status()
            verdict = parse_verdict(resp.json()["message"]["content"], self.name)
            if verdict is not None:
                return verdict
        return None


class CodexJudge:
    """Cross-check judge: gpt-5.5 through the Codex CLI, read-only sandbox."""

    def __init__(self, binary: str = "codex"):
        self.binary = binary
        self.name = "codex:gpt-5.5"

    def judge(self, prompt: str, retries: int = 1) -> Verdict | None:
        full_prompt = JUDGE_SYSTEM + "\n\n" + prompt
        for _ in range(retries + 1):
            try:
                proc = subprocess.run(
                    [self.binary, "exec", "-s", "read-only", "--skip-git-repo-check", full_prompt],
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return None
            verdict = parse_verdict(proc.stdout, self.name)
            if verdict is not None:
                return verdict
        return None


def make_judge(spec: str):
    """Build a judge from a spec string: 'ollama:<model>' or 'codex'."""
    if spec == "codex":
        return CodexJudge()
    if spec.startswith("ollama:"):
        return OllamaJudge(model=spec.split(":", 1)[1])
    raise ValueError(f"Unknown judge spec: {spec}")


# ---------------------------------------------------------------------------
# Judge agreement
# ---------------------------------------------------------------------------


def cohen_kappa(a: list[int], b: list[int], weighted: bool = False) -> float:
    """Cohen's kappa for two raters; linear weights for the ordinal 0/1/2 scale."""
    assert len(a) == len(b) and a, "need paired non-empty label lists"
    categories = sorted(set(a) | set(b))
    n = len(a)
    index = {c: i for i, c in enumerate(categories)}
    m = len(categories)
    obs = [[0.0] * m for _ in range(m)]
    for x, y in zip(a, b):
        obs[index[x]][index[y]] += 1 / n
    pa = [sum(1 for x in a if x == c) / n for c in categories]
    pb = [sum(1 for y in b if y == c) / n for c in categories]
    if weighted:
        w = [[1 - abs(i - j) / (m - 1) if m > 1 else 1.0 for j in range(m)] for i in range(m)]
    else:
        w = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    po = sum(w[i][j] * obs[i][j] for i in range(m) for j in range(m))
    pe = sum(w[i][j] * pa[i] * pb[j] for i in range(m) for j in range(m))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
