# Employee RAG Evaluation Report

Run ID: run-20260609T115029Z

Release: employee-rag-v1.0.0

Manifest: `manifests/employee-rag-v1.0.0.yaml`

Threshold validation: PASS

Scoring method: RAGAS LLM-as-judge metrics

## Configuration

Model:
- meta/llama-3.3-70b-instruct

Embeddings:
- voyage-3-large

Vector Store:
- chromadb

Dataset:
- 5 evaluation questions

## Retrieval Evaluation

| Metric | Score |
| --- | ---: |
| Context Precision | 0.99 |
| Context Recall | 1.0 |
| Context Relevance | 0.99 |

## Generation Evaluation

| Metric | Score |
| --- | ---: |
| Faithfulness | 0.95 |
| Response Relevance | 0.5674 |

## RAG System Evaluation

| Metric | Score |
| --- | ---: |
| Completeness | 0.7617 |
| Hallucination Rate | 0.05 |

## Performance

| Metric | Value |
| --- | ---: |
| Avg Retrieval Latency | 570.196 ms |
| Avg Generation Latency | 28894.46 ms |
| Avg Total Latency | 29464.654 ms |
| Avg Prompt Tokens | 642.4 |
| Avg Completion Tokens | 23.4 |
| Avg Total Tokens | 665.8 |

## Gate Validation

Overall status: FAIL

Metric coverage: 5 complete, 0 partial, 0 failed

Gate result: 3 passed, 2 failed, 0 not evaluated

| Gate | Actual | Target | Rule | Status |
| --- | ---: | ---: | --- | --- |
| Context Relevance | 0.99 | 0.8 | actual >= target | PASS |
| Faithfulness | 0.95 | 0.85 | actual >= target | PASS |
| Response Relevance | 0.5674 | 0.85 | actual >= target | FAIL |
| Completeness | 0.7617 | 0.8 | actual >= target | FAIL |
| Hallucination Rate | 0.05 | 0.1 | actual <= target | PASS |

## Why Gates Failed

- `response_relevance`: 0.5674 is below the required threshold of 0.85. Action: Review RAG retrieval, generation prompt, and evaluator configuration.
- `completeness`: 0.7617 is below the required threshold of 0.8. Action: Review RAG retrieval, generation prompt, and evaluator configuration.

## Metric Coverage

| Metric | Scored | Total | Status |
| --- | ---: | ---: | --- |
| Faithfulness | 5 | 5 | COMPLETE |
| Response Relevance | 5 | 5 | COMPLETE |
| Completeness | 5 | 5 | COMPLETE |
| Context Precision | 5 | 5 | COMPLETE |
| Context Recall | 5 | 5 | COMPLETE |
