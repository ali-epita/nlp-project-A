# Retrieval versus generation: locating the bottleneck

Model: `qwen2.5:14b-instruct-q4_K_M`, 150 questions, correctness judged by the primary judge.

| Outcome | Count | Share | Meaning |
|---|---|---|---|
| Solved | 43 | 28.7% | correct with retrieved and with gold evidence |
| Retrieval-limited | 46 | 30.7% | correct with gold evidence, wrong with retrieved |
| Generation-limited | 61 | 40.7% | wrong even with gold evidence |
| Retrieval-rescued | 12 | 8.0% | wrong with gold evidence, right with retrieved |

The retrieval-limited share is the headroom a better retriever could
recover; the generation-limited share is the ceiling of the model itself.
A large retrieval-rescued share would suggest answers arriving from
pre-training rather than the corpus and should be read against the
closed-book run.

## Generation-limited questions by reasoning type

| Reasoning | Count |
|---|---|
| unspecified | 22 |
| Numerical reasoning | 14 |
| Information extraction | 14 |
| Logical reasoning (based on numerical reasoning) | 11 |

## Retrieval-limited questions by reasoning type

| Reasoning | Count |
|---|---|
| Numerical reasoning | 21 |
| unspecified | 13 |
| Information extraction | 10 |
| Logical reasoning (based on numerical reasoning) | 2 |