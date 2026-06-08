# RAG Evaluation Report

Run ID: run-20260608T153814Z

Release: employee-rag-v1.0.0

Manifest: `manifests/employee-rag-v1.0.0.yaml`

Threshold validation: pass

Scoring method: offline lexical proxy metrics; no Groq or Voyage judge/scoring calls

## Scores

| Metric | Score | Gate | Status |
| --- | ---: | ---: | --- |
| Faithfulness | 0.8789 | 0.85 | pass |
| Response Relevance | 0.681 | 0.85 | fail |
| Completeness | 0.662 | 0.8 | fail |
| Context Relevance | 0.76 | 0.8 | fail |
| Context Recall | 1.0 | 0.8 | pass |
| Hallucination Rate | 0.1211 | 0.1 | fail |

## Performance

| Metric | Value |
| --- | ---: |
| Avg Retrieval Latency | 447.696 ms |
| Avg Generation Latency | 598.3 ms |
| Avg Total Latency | 1045.998 ms |
| Avg Prompt Tokens | 685.2 |
| Avg Completion Tokens | 119.2 |
| Avg Total Tokens | 804.4 |
