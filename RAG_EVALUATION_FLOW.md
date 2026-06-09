# RAG Evaluation Flow

This document explains how the project evaluates the RAG system.

## Short Answer

The evaluator does not evaluate only the question text.

It evaluates the saved output of the full RAG pipeline:

```text
question
generated answer
retrieved contexts
ground truth answer
latency
token usage
```

Those saved RAG outputs are stored in:

```text
reports/raw_rag_answers/
```

## End-to-End Flow

```text
evaluation/qa_dataset.json
        |
        v
python evaluation/generate_answers.py
        |
        v
Run real RAG pipeline
        |
        v
Retrieve chunks from ChromaDB
        |
        v
Generate answer using NVIDIA Llama 3.3 70B Instruct
        |
        v
Save raw RAG outputs
        |
        v
reports/raw_rag_answers/Q001.json ... Q005.json
        |
        v
python evaluation/ragas_eval.py
        |
        v
Calculate RAGAS evaluation scores
        |
        v
reports/raga_eval/<run_id>/result.json
reports/raga_eval/<run_id>/result.yaml
reports/raga_eval/<run_id>/scorecard.md
reports/raga_eval/<run_id>/scorecard.json
```

## Step 1: Dataset

The evaluation starts with:

```text
evaluation/qa_dataset.json
```

Example:

```json
{
  "id": "Q001",
  "question": "How many annual leave days are employees entitled to?",
  "expected_answer": "Employees receive 24 days of annual leave per year."
}
```

This file contains the test questions and expected answers.

## Step 2: Generate Raw RAG Answers

Command:

```bash
python evaluation/generate_answers.py
```

This runs the actual RAG system.

For each question:

```text
question
   |
   v
Voyage query embedding
   |
   v
ChromaDB retrieval
   |
   v
top_k handbook chunks
   |
   v
NVIDIA Llama 3.3 70B Instruct answer generation
   |
   v
raw answer record
```

Output:

```text
reports/raw_rag_answers/Q001.json
reports/raw_rag_answers/Q002.json
reports/raw_rag_answers/Q003.json
reports/raw_rag_answers/Q004.json
reports/raw_rag_answers/Q005.json
```

Each raw answer file contains:

```text
id
question
answer
retrieved contexts
sources
ground_truth
retrieval_time_ms
generation_time_ms
total_latency_ms
input_tokens
output_tokens
total_tokens
```

So this stage answers:

```text
What did our RAG system actually produce?
```

## Step 3: Evaluate RAG Outputs

Command:

```bash
python evaluation/ragas_eval.py
```

This reads:

```text
reports/raw_rag_answers/
```

and the internal combined cache:

```text
evaluation/results/answer_records.json
```

Then it runs real RAGAS metrics using:

```python
from ragas import evaluate
```

RAGAS uses an evaluator LLM and embeddings for semantic scoring. In this project, the evaluator is configured with NVIDIA OpenAI-compatible chat completions and Voyage embeddings.

## What Gets Evaluated

### Retrieval Quality

Checks whether the retrieved handbook chunks are useful.

Metrics:

```text
context_relevance
context_recall
```

Meaning:

```text
Did retrieval bring back chunks that contain the information needed to answer?
```

### Generation Quality

Checks whether the generated answer matches the expected answer and is supported by context.

Metrics:

```text
faithfulness
response_relevance
completeness
hallucination_rate
```

Meaning:

```text
Did the LLM answer correctly using the retrieved context?
```

### Performance

Tracks runtime and token usage.

Metrics:

```text
avg_retrieval_latency_ms
avg_generation_latency_ms
avg_total_latency_ms
avg_prompt_tokens
avg_completion_tokens
avg_total_tokens
```

Meaning:

```text
How expensive and slow was the RAG system?
```

## Gates

The thresholds are defined in:

```text
manifests/employee-rag-v1.0.0.yaml
```

Current gates:

```yaml
gates:
  context_relevance: 0.80
  faithfulness: 0.85
  response_relevance: 0.85
  completeness: 0.80
  hallucination_rate: 0.10
```

The evaluator compares RAGAS scores against these gates.

Example:

```text
faithfulness score >= 0.85 -> pass
faithfulness score < 0.85  -> fail
```

For hallucination rate:

```text
hallucination_rate <= 0.10 -> pass
hallucination_rate > 0.10  -> fail
```

## Final Evaluation Outputs

The evaluation report is stored in:

```text
reports/raga_eval/
```

Files:

```text
reports/raga_eval/<run_id>/result.json
reports/raga_eval/<run_id>/result.yaml
reports/raga_eval/<run_id>/scorecard.md
reports/raga_eval/<run_id>/scorecard.json
```

`result.json` and `result.yaml` are detailed machine-readable results.

`scorecard.md` is the clean human-readable summary.

## Summary

The evaluation is not checking only questions.

It checks the full RAG output:

```text
question
answer
retrieved context
expected answer
latency
tokens
```

The raw RAG answer files are the bridge between:

```text
running the RAG system
```

and:

```text
evaluating the RAG system
```
