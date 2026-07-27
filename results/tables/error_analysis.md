# Retrieval Error Analysis

Best configuration with k <= 10: **filter@k=10** (recall@10 = 0.627 over 150 questions)

## Failure taxonomy

| Failure mode | Count | Share of questions |
|---|---|---|
| Wrong document retrieved | 3 | 2.0% |
| Right document, wrong page | 53 | 35.3% |
| Gold page hit but evidence text mostly absent | 19 | 12.7% |
| **Total misses at k=10** | **56** | 37.3% |

Wrong-document failures are discrimination errors (company, year, or filing
type not pinned down) and respond to metadata filtering. Right-document
wrong-page failures are ranking errors inside one filing and respond to
chunking and reranking. The third row only shows up in text-level measures:
the page was retrieved but the chunk boundary cut the evidence away.

## Failures by question type

| Question type | Failures | Total | Failure rate |
|---|---|---|---|
| domain-relevant | 28 | 50 | 56.0% |
| metrics-generated | 8 | 50 | 16.0% |
| novel-generated | 20 | 50 | 40.0% |

## Individual failures

| id | type | doc found | coverage |
|---|---|---|---|
| financebench_id_00669 | domain-relevant | no | 0.00 |
| financebench_id_01911 | novel-generated | no | 0.05 |
| financebench_id_00601 | novel-generated | no | 0.05 |
| financebench_id_03029 | metrics-generated | yes | 0.40 |
| financebench_id_04672 | metrics-generated | yes | 0.01 |
| financebench_id_00499 | domain-relevant | yes | 0.05 |
| financebench_id_01226 | domain-relevant | yes | 0.01 |
| financebench_id_01865 | novel-generated | yes | 0.00 |
| financebench_id_00807 | domain-relevant | yes | 0.04 |
| financebench_id_00941 | domain-relevant | yes | 0.00 |
| financebench_id_00799 | domain-relevant | yes | 0.00 |
| financebench_id_01079 | domain-relevant | yes | 0.01 |
| financebench_id_00684 | domain-relevant | yes | 0.10 |
| financebench_id_01936 | novel-generated | yes | 0.00 |
| financebench_id_01928 | novel-generated | yes | 0.03 |
| financebench_id_01930 | novel-generated | yes | 0.00 |
| financebench_id_00222 | domain-relevant | yes | 0.00 |
| financebench_id_00563 | novel-generated | yes | 0.38 |
| financebench_id_00757 | novel-generated | yes | 0.12 |
| financebench_id_01028 | domain-relevant | yes | 0.02 |
| financebench_id_00723 | domain-relevant | yes | 0.08 |
| financebench_id_00720 | domain-relevant | yes | 0.01 |
| financebench_id_01351 | domain-relevant | yes | 0.03 |
| financebench_id_01964 | novel-generated | yes | 0.01 |
| financebench_id_01981 | novel-generated | yes | 0.00 |
| financebench_id_00685 | domain-relevant | yes | 0.04 |
| financebench_id_01077 | domain-relevant | yes | 0.10 |
| financebench_id_01275 | domain-relevant | yes | 0.07 |
| financebench_id_00288 | novel-generated | yes | 0.01 |
| financebench_id_07661 | metrics-generated | yes | 0.04 |
| financebench_id_01290 | domain-relevant | yes | 0.00 |
| financebench_id_00464 | novel-generated | yes | 0.95 |
| financebench_id_00585 | novel-generated | yes | 0.34 |
| financebench_id_00005 | domain-relevant | yes | 0.03 |
| financebench_id_04103 | metrics-generated | yes | 0.09 |
| financebench_id_00956 | domain-relevant | yes | 0.02 |
| financebench_id_00711 | domain-relevant | yes | 0.02 |
| financebench_id_01484 | novel-generated | yes | 0.03 |
| financebench_id_01487 | novel-generated | yes | 0.17 |
| financebench_id_02119 | novel-generated | yes | 0.00 |
| financebench_id_00206 | domain-relevant | yes | 0.00 |
| financebench_id_04171 | metrics-generated | yes | 0.03 |
| financebench_id_00382 | novel-generated | yes | 0.03 |
| financebench_id_04700 | metrics-generated | yes | 0.03 |
| financebench_id_00552 | domain-relevant | yes | 0.01 |
| financebench_id_04302 | metrics-generated | yes | 0.17 |
| financebench_id_00080 | domain-relevant | yes | 0.08 |
| financebench_id_01009 | domain-relevant | yes | 0.11 |
| financebench_id_00735 | domain-relevant | yes | 0.04 |
| financebench_id_01328 | domain-relevant | yes | 0.19 |
| financebench_id_00302 | novel-generated | yes | 0.03 |
| financebench_id_00724 | novel-generated | yes | 0.17 |
| financebench_id_00521 | domain-relevant | yes | 0.04 |
| financebench_id_02024 | novel-generated | yes | 0.01 |
| financebench_id_00216 | domain-relevant | yes | 0.03 |
| financebench_id_04784 | metrics-generated | yes | 0.06 |

## Questions excluded by the metadata filter analysis

Companies parsed from question text cover the filter runs; see the
strategy figure for the filter's effect at matched k.