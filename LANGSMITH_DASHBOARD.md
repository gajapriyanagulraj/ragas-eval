# LangSmith Observability Dashboard

This project uses LangSmith as an optional observability layer on top of the existing RAG and RAGAS pipeline.

The core RAG system still runs locally:

```text
qa_dataset.json
  -> ChromaDB retrieval
  -> NVIDIA Llama 3.3 70B Instruct
  -> raw RAG answers
  -> RAGAS evaluation
  -> result + scorecard
```

LangSmith adds:

```text
traces
metadata
RAGAS feedback scores
dataset view
experiment-level visibility
```

## Enable LangSmith

Create a LangSmith API key and add these values to `.env`:

```env
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=employee-rag-evaluation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## What Gets Sent To LangSmith

### RAG Trace

Each question creates a LangSmith run named:

```text
employee_handbook_rag
```

Inputs:

```text
question
```

Outputs:

```text
generated answer
retrieved contexts
sources
```

Metadata:

```text
release_id
manifest path
question_id
contexts_retrieved
retrieval_latency_ms
generation_latency_ms
total_latency_ms
prompt_tokens
completion_tokens
total_tokens
model
embedding_model
chunk_size
chunk_overlap
top_k
```

## RAGAS Feedback Scores

After `ragas.evaluate(...)` completes, the evaluation script attaches feedback scores to the matching LangSmith run.

Feedback keys:

```text
context_precision
context_recall
faithfulness
response_relevance
completeness
hallucination_rate
```

These are the same values written to:

```text
reports/raga_eval/<run_id>/result.json
reports/raga_eval/<run_id>/result.yaml
```

## Evaluation Summary Run

Each execution of:

```bash
python evaluation/ragas_eval.py
```

also creates one summary run named:

```text
employee_rag_ragas_evaluation
```

This run represents the overall report.

Inputs:

```text
release_id
questions_evaluated
```

Outputs:

```text
overall status
average retrieval metrics
average generation metrics
average RAG system metrics
average latency
average token usage
quality gates
```

Metadata:

```text
evaluation_run_id
result.json path
scorecard.json path
manifest path
release status
```

Feedback scores:

```text
average context_precision
average context_recall
average faithfulness
average response_relevance
average completeness
average hallucination_rate
```

## Dataset View

The evaluation script creates or updates this LangSmith dataset:

```text
Employee Handbook Evaluation
```

Each example contains:

```text
question id
question
expected answer
generated answer
question status
release id
evaluation run id
scorecard path
```

## Dashboard Sections

Use LangSmith to inspect:

```text
Quality
  faithfulness
  response_relevance
  completeness
  hallucination_rate

Retrieval
  context_precision
  context_recall

Performance
  retrieval latency
  generation latency
  total latency
  prompt tokens
  completion tokens
  total tokens

Governance
  release_id
  manifest
  scorecard
  evaluation run id
```

## Run Flow

Fresh answer generation:

```bash
python evaluation/generate_answers.py --force
```

RAGAS evaluation and feedback publishing:

```bash
python evaluation/ragas_eval.py
```

Then open LangSmith project:

```text
employee-rag-evaluation
```

You should see:

```text
Experiment / Project
├── employee_rag_ragas_evaluation
├── Q001 RAG trace
├── Q002 RAG trace
├── Q003 RAG trace
├── Q004 RAG trace
└── Q005 RAG trace
```

Each trace contains the raw RAG call, metadata, and RAGAS feedback scores.

## Local Reports Still Remain Source Of Record

LangSmith is for observability and visualization.

The release artifacts remain in the repository:

```text
manifests/employee-rag-v1.0.0.yaml
reports/raga_eval/<run_id>/scorecard.json
reports/raga_eval/<run_id>/result.json
```
