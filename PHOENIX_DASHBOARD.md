# Arize Phoenix Observability

This project uses Arize Phoenix as an optional observability layer for the existing RAG and RAGAS pipeline.

The core pipeline remains:

```text
qa_dataset.json
  -> ChromaDB retrieval
  -> NVIDIA Llama 3.3 70B Instruct
  -> raw RAG answers
  -> RAGAS evaluation
  -> result + scorecard
```

Phoenix adds trace visualization for:

```text
RAG question runs
retrieval metadata
generation latency
token usage
RAGAS question scores
overall evaluation summary
```

## Enable Phoenix

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Phoenix locally:

```bash
phoenix serve
```

Phoenix normally opens at:

```text
http://localhost:6006
```

Add these values to `.env`:

```env
PHOENIX_ENABLED=true
PHOENIX_PROJECT_NAME=employee-rag-evaluation
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
```

Verify Phoenix receives traces:

```bash
python evaluation/phoenix_smoke_test.py
```

Then open Phoenix and search for:

```text
employee_handbook_rag
```

## What Gets Traced

### RAG Question Trace

Each dataset question creates a span named:

```text
employee_handbook_rag
```

Attributes include:

```text
release_id
manifest
question_id
input.question
output.answer
retrieval.contexts_retrieved
retrieval.latency_ms
generation.latency_ms
performance.total_latency_ms
tokens.prompt
tokens.completion
tokens.total
model
embedding_model
chunk_size
chunk_overlap
top_k
```

### RAGAS Question Evaluation

Each question also creates a span named:

```text
ragas_question_evaluation
```

Attributes include:

```text
run_id
evaluation_run_id
question_id
ragas.context_precision
ragas.context_recall
ragas.faithfulness
ragas.response_relevance
ragas.completeness
ragas.hallucination_rate
```

### Evaluation Summary

Each evaluation run creates a span named:

```text
employee_rag_ragas_evaluation
```

Attributes include:

```text
run_id
status
result
scorecard
questions_evaluated
average retrieval scores
average generation scores
average RAG system scores
average latency
average token usage
```

## Filter By Run ID

Use the evaluation run ID in Phoenix filters.

Example:

```text
run-20260609T170918Z
```

The value is stored as:

```text
run_id
evaluation_run_id
```

Use this to isolate one evaluation execution.

## Run Flow

Fresh answer generation:

```bash
python evaluation/generate_answers.py --force
```

RAGAS evaluation:

```bash
python evaluation/ragas_eval.py
```

Then open:

```text
http://localhost:6006
```

You should see spans for:

```text
employee_handbook_rag
ragas_question_evaluation
employee_rag_ragas_evaluation
```

## Local Reports Remain Source Of Record

Phoenix is for observability and visualization.

The release artifacts remain:

```text
manifests/employee-rag-v1.0.0.yaml
reports/raga_eval/<run_id>/result.json
reports/raga_eval/<run_id>/scorecard.json
```
