# End-to-End RAG Evaluation

This document explains how this project evaluates the RAG system from dataset question to final RAGAS scorecard.

## What Is Being Evaluated

The evaluation is not scoring only the question text.

It evaluates the full RAG output:

```text
question
retrieved contexts
generated answer
expected answer
latency
token usage
```

That means the evaluation checks both parts of the RAG system:

```text
retrieval quality + generation quality
```

## Full Flow

```text
evaluation/qa_dataset.json
        |
        v
python evaluation/generate_answers.py
        |
        v
For each question:
  1. Embed the question with Voyage
  2. Search ChromaDB
  3. Retrieve top_k handbook chunks
  4. Send question + contexts to NVIDIA Llama 3.3 70B Instruct
  5. Save the generated answer
        |
        v
reports/raw_rag_answers/Q001.json ... Q005.json
        |
        v
python evaluation/ragas_eval.py
        |
        v
RAGAS evaluates:
  question + generated answer + contexts + ground truth
        |
        v
reports/raga_eval/<run_id>/result.json
reports/raga_eval/<run_id>/result.yaml
reports/raga_eval/<run_id>/scorecard.json
reports/raga_eval/<run_id>/scorecard.md
```

## Step 1: Evaluation Dataset

The test questions are stored here:

```text
evaluation/qa_dataset.json
```

Each record contains:

```json
{
  "id": "Q001",
  "question": "How many annual leave days are employees entitled to?",
  "expected_answer": "Employees receive 24 days of annual leave per year."
}
```

The raw answer file keeps this value as `ground_truth`. When passed into RAGAS, it is mapped to RAGAS's `reference` column.

## Step 2: Generate Raw RAG Answers

Run:

```bash
python evaluation/generate_answers.py
```

This runs the real RAG system.

For each dataset question:

```text
Question
   |
   v
Voyage embedding
   |
   v
ChromaDB similarity search
   |
   v
Top-K retrieved contexts
   |
   v
NVIDIA Llama 3.3 70B Instruct generation
   |
   v
Raw answer JSON
```

Example output:

```text
reports/raw_rag_answers/Q001.json
```

That file contains:

```json
{
  "id": "Q001",
  "question": "...",
  "answer": "...",
  "contexts": ["...", "..."],
  "sources": ["employee_handbook.txt"],
  "ground_truth": "...",
  "retrieval_time_ms": 528.32,
  "generation_time_ms": 901.18,
  "total_latency_ms": 1429.5,
  "input_tokens": 721,
  "output_tokens": 127,
  "total_tokens": 848
}
```

This file answers:

```text
What did our RAG system actually retrieve and generate?
```

## Step 3: Run RAGAS Evaluation

Run:

```bash
python evaluation/ragas_eval.py
```

This file uses actual RAGAS:

```python
from ragas import evaluate
```

It builds a RAGAS dataset from the raw answer records.

Raw answer fields are mapped to the RAGAS 0.4.x schema:

```text
question     -> user_input
answer       -> response
contexts     -> retrieved_contexts
ground_truth -> reference
```

The dataset passed to RAGAS looks like:

```python
{
    "user_input": question,
    "response": generated_answer,
    "retrieved_contexts": retrieved_contexts,
    "reference": expected_answer
}
```

Then it runs:

```python
evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        answer_correctness,
        context_precision,
        context_recall,
    ],
)
```

## What Each RAGAS Metric Means

### Context Precision

Checks whether the retrieved chunks are relevant.

```text
High score = retrieved chunks are useful for answering the question
Low score = retrieval returned unnecessary or unrelated chunks
```

Used in the report as:

```text
context_relevance
```

### Context Recall

Checks whether the retrieved chunks contain the information needed to match the ground truth.

```text
High score = retrieval found the needed information
Low score = retrieval missed important information
```

### Faithfulness

Checks whether the generated answer is supported by the retrieved contexts.

```text
High score = answer is grounded in retrieved context
Low score = answer says things not supported by context
```

### Answer Relevancy

Checks whether the generated answer directly answers the question.

```text
High score = answer is focused on the question
Low score = answer is incomplete, indirect, or off-topic
```

Used in the report as:

```text
response_relevance
```

### Answer Correctness

Checks whether the generated answer semantically matches the expected answer.

```text
High score = answer matches the ground truth
Low score = answer misses or changes the expected meaning
```

Used in the report as:

```text
completeness
```

### Hallucination Rate

This is derived from faithfulness:

```text
hallucination_rate = 1 - faithfulness
```

Example:

```text
faithfulness = 0.94
hallucination_rate = 0.06
```

## Step 4: Apply Manifest Gates

The thresholds are defined here:

```text
manifests/employee-rag-v1.0.0.yaml
```

Current gates:

```yaml
gates:
  min_context_relevance: 0.80
  min_faithfulness: 0.85
  min_response_relevance: 0.85
  min_completeness: 0.80
  max_hallucination_rate: 0.10
```

Pass/fail rules:

```text
faithfulness >= 0.85       -> pass
response_relevance >= 0.85 -> pass
completeness >= 0.80       -> pass
context_relevance >= 0.80  -> pass
context_recall >= 0.80     -> pass
hallucination_rate <= 0.10 -> pass
```

Hallucination rate is the only inverted gate because lower is better.

## Step 5: Final Reports

RAGAS evaluation writes reports here:

```text
reports/raga_eval/
```

### Detailed Result

```text
reports/raga_eval/<run_id>/result.json
reports/raga_eval/<run_id>/result.yaml
```

These contain:

```text
run_id
release_id
question
expected answer
generated answer
per-question retrieval scores
per-question generation scores
per-question RAG system scores
per-question performance
per-question status
```

Each question is grouped like this:

```json
{
  "id": "Q001",
  "question": "...",
  "expected_answer": "...",
  "generated_answer": "...",
  "retrieval": {
    "context_precision": 1.0,
    "context_recall": 1.0
  },
  "generation": {
    "faithfulness": 1.0,
    "response_relevance": 0.5417
  },
  "rag_system": {
    "completeness": 0.7322,
    "hallucination_rate": 0.0
  },
  "performance": {
    "latency_ms": 1429.5,
    "prompt_tokens": 721,
    "completion_tokens": 127,
    "total_tokens": 848
  },
  "status": "FAIL"
}
```

### Scorecard

```text
reports/raga_eval/<run_id>/scorecard.json
reports/raga_eval/<run_id>/scorecard.md
```

The scorecard is the clean summary:

```text
total questions
overall scores
pass/fail gate result
why a gate failed
average latency
average token usage
```

## How To Run From Start To Finish

Use this when you want a fresh evaluation:

```bash
source .venv/bin/activate
python src/ingest.py
python evaluation/generate_answers.py
python evaluation/ragas_eval.py
cat reports/raga_eval/<run_id>/scorecard.md
```

Run `python src/ingest.py` only when the handbook, chunk size, chunk overlap, embedding model, or ChromaDB collection changes.

For normal evaluation reruns, use:

```bash
python evaluation/generate_answers.py
python evaluation/ragas_eval.py
```

## One-Line Summary

```text
qa_dataset.json -> real RAG answers -> raw answer files -> RAGAS semantic scoring -> scorecard
```
