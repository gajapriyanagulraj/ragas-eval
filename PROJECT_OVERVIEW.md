# Employee Handbook RAG Evaluation Project

This project implements a complete Retrieval-Augmented Generation workflow and a RAGAS evaluation pipeline for an employee handbook assistant.

## Objective

Build a simple RAG system that can answer employee handbook questions, then evaluate the RAG output using a reproducible manifest-driven workflow.

## Architecture

```text
                         data/employee_handbook.txt
                                   |
                                   v
                              Document Loader
                                   |
                                   v
                                Chunking
                         chunk_size=800, overlap=150
                                   |
                                   v
                           Voyage AI Embeddings
                              voyage-3-large
                                   |
                                   v
                                ChromaDB
                         collection=employee_handbook
                                   |
                                   v
User Question ---> Query Embedding ---> Similarity Search top_k=5
                                   |
                                   v
                            Retrieved Contexts
                                   |
                                   v
                          NVIDIA Llama 3.3 70B Instruct
                                   |
                                   v
                          Answer + Sources
                                   |
                                   v
                         Raw RAG Answer Files
                                   |
                                   v
                            RAGAS Evaluation
                                   |
                                   v
                     Result JSON/YAML + Scorecard
```

## Main Components

### RAG Application

The RAG application answers questions using the employee handbook.

Input:

```text
User question
```

Process:

```text
Retrieve relevant handbook chunks from ChromaDB
Send retrieved chunks to NVIDIA Llama 3.3 70B Instruct
Generate grounded answer
Return answer and source
```

Key files:

```text
src/ingest.py
src/retriever.py
src/llm.py
src/rag.py
ui/app.py
```

## Ingestion

Run ingestion only when the handbook, chunking, or embedding settings change.

```bash
python src/ingest.py
```

This creates or refreshes the ChromaDB vector index:

```text
chroma_db/
```

## Evaluation Dataset

The evaluation questions are stored in JSON:

```text
evaluation/qa_dataset.json
```

The dataset contains five questions and expected answers.

## Manifest

The manifest freezes the experiment configuration:

```text
manifests/employee-rag-v1.0.0.yaml
```

It defines:

```text
model
dataset
embedding model
vector store
chunk size
chunk overlap
top_k
evaluation gates
```

The gate thresholds are:

```yaml
gates:
  min_context_relevance: 0.80
  min_faithfulness: 0.85
  min_response_relevance: 0.85
  min_completeness: 0.80
  max_hallucination_rate: 0.10
```

The evaluator validates that all gate values are numeric values between 0 and 1.

## Evaluation Flow

### Step 1: Generate Raw RAG Answers

```bash
python evaluation/generate_answers.py
```

This runs the real RAG system against the dataset questions.

Outputs:

```text
reports/raw_rag_answers/Q001.json
reports/raw_rag_answers/Q002.json
reports/raw_rag_answers/Q003.json
reports/raw_rag_answers/Q004.json
reports/raw_rag_answers/Q005.json
```

Each file contains:

```text
question
answer
retrieved contexts
sources
ground truth
retrieval latency
generation latency
token usage
```

### Step 2: Run RAGAS Evaluation

```bash
python evaluation/ragas_eval.py
```

This reads the saved raw RAG answers and runs real RAGAS metrics with `ragas.evaluate(...)`.

RAGAS uses an evaluator LLM and embeddings for semantic scoring.

Scores:

```text
faithfulness
response relevance
completeness
context relevance
context recall
hallucination rate
```

Outputs:

```text
reports/raga_eval/<run_id>/result.json
reports/raga_eval/<run_id>/result.yaml
reports/raga_eval/<run_id>/scorecard.md
reports/raga_eval/<run_id>/scorecard.json
```

## Result vs Scorecard

`result.json` and `result.yaml` are detailed per-question machine-readable outputs.

They include:

```text
run_id
release_id
question
expected answer
generated answer
per-question retrieval scores
per-question generation scores
per-question RAG system scores
per-question latency
per-question token usage
per-question status
```

`scorecard.json` and `scorecard.md` are the clean release-level summaries for review or demo.

They include:

```text
run_id
release_id
overall status
average metric scores
average latency
average token usage
quality gates
release decision
blocking issues
```

## Quality Gate

Each score is compared against the manifest gate.

Example:

```text
faithfulness score >= faithfulness gate -> pass
faithfulness score < faithfulness gate  -> fail
```

Hallucination rate is inverted:

```text
hallucination rate <= hallucination gate -> pass
hallucination rate > hallucination gate  -> fail
```

## Standard Run Procedure

```bash
source .venv/bin/activate
python evaluation/generate_answers.py
python evaluation/ragas_eval.py
cat reports/raga_eval/<run_id>/scorecard.md
```

## When To Rerun Each Step

Rerun ingestion when changing:

```text
employee_handbook.txt
chunk_size
chunk_overlap
embedding_model
```

Rerun raw answer generation when changing:

```text
qa_dataset.json
prompt
top_k
LLM settings
retrieval settings
handbook index
```

Rerun evaluation when changing:

```text
gates
RAGAS scoring configuration
report format
```

## Final Output Structure

```text
reports/
├── raw_rag_answers/
│   ├── Q001.json
│   ├── Q002.json
│   ├── Q003.json
│   ├── Q004.json
│   └── Q005.json
│
└── raga_eval/
    └── <run_id>/
        ├── result.json
        ├── result.yaml
        ├── scorecard.json
        └── scorecard.md
```

## Summary

This project now works as:

```text
RAG app:
handbook -> embeddings -> ChromaDB -> retrieved context -> NVIDIA Llama 3.3 70B Instruct answer

Evaluation:
dataset -> raw RAG answers -> RAGAS scores -> pass/fail scorecard

Governance:
manifest -> fixed config + thresholds -> reproducible evaluation
```
