# RAGAS Evaluation Architecture

This document shows the end-to-end architecture for the employee handbook RAG evaluation pipeline.

## System Mindmap

```mermaid
mindmap
  root((Employee Handbook RAG Evaluation))
    Knowledge Base
      employee_handbook.txt
      Chunking
        chunk_size 800
        chunk_overlap 150
      Embeddings
        Voyage AI
        voyage-3-large
      Vector Store
        ChromaDB
        collection employee_handbook
    RAG Application
      Dataset Questions
        evaluation/qa_dataset.json
        5 questions
      Retrieval
        query embedding
        top_k 5
        retrieved contexts
      Generation
        NVIDIA Llama 3.3 70B Instruct
        grounded answer
        sources
      Raw Outputs
        reports/raw_rag_answers/Q001.json
        evaluation/results/answer_records.json
    RAGAS Evaluation
      RAGAS evaluate
      Evaluator LLM
        NVIDIA Llama 3.3 70B Instruct
      Evaluator Embeddings
        Voyage AI
      Metrics
        context_precision
        context_recall
        faithfulness
        answer_relevancy
        answer_correctness
    LangSmith Observability
      Project
        employee-rag-evaluation
      Traces
        Q001-Q005 RAG calls
      Feedback
        RAGAS metric scores
      Dataset
        Employee Handbook Evaluation
    Reporting
      Per Question Result
        result.json
        result.yaml
      Release Scorecard
        scorecard.json
        scorecard.md
      Quality Gates
        PASS
        FAIL
        release_decision
    Governance
      Manifest
        employee-rag-v1.0.0.yaml
      Frozen Config
        model
        embedding model
        chunking
        top_k
        dataset
        gates
```

## End-to-End Flow

```mermaid
flowchart TD
    A[evaluation/qa_dataset.json] --> B[generate_answers.py]
    B --> C[HandbookRAG]
    C --> D[Voyage query embedding]
    D --> E[ChromaDB similarity search]
    E --> F[Top-K retrieved handbook contexts]
    F --> G[NVIDIA Llama 3.3 70B Instruct]
    G --> H[Generated answer]
    H --> I[Raw RAG answer records]
    F --> I
    I --> J[reports/raw_rag_answers/Q001-Q005.json]
    I --> K[evaluation/results/answer_records.json]

    K --> L[ragas_eval.py]
    L --> M[RAGAS dataset]
    M --> N[ragas.evaluate]
    N --> O[RAGAS metrics]
    O --> P[metrics.py report builder]
    O --> V[LangSmith feedback scores]
    I --> W[LangSmith RAG traces]
    P --> Q[reports/raga_eval/run_id/result.json]
    P --> R[reports/raga_eval/run_id/result.yaml]
    P --> S[reports/raga_eval/run_id/scorecard.json]
    P --> T[reports/raga_eval/run_id/scorecard.md]
    P --> X[LangSmith dataset view]

    U[manifests/employee-rag-v1.0.0.yaml] --> B
    U --> L
    U --> P
```

## Runtime Sequence

```mermaid
sequenceDiagram
    participant Dataset as qa_dataset.json
    participant Gen as generate_answers.py
    participant RAG as HandbookRAG
    participant VS as ChromaDB
    participant LLM as NVIDIA Llama 3.3 70B
    participant Raw as Raw Answer Records
    participant Eval as ragas_eval.py
    participant RAGAS as RAGAS evaluate()
    participant Report as Reports

    Dataset->>Gen: Load 5 questions and expected answers
    Gen->>RAG: Ask each question
    RAG->>VS: Retrieve top_k contexts using Voyage embeddings
    VS-->>RAG: Return handbook chunks
    RAG->>LLM: Send question + retrieved contexts
    LLM-->>RAG: Return generated answer
    RAG-->>Gen: Answer, contexts, latency, tokens
    Gen->>Raw: Save raw answer records

    Eval->>Raw: Load question, answer, contexts, ground_truth
    Eval->>RAGAS: Evaluate with RAGAS metrics
    RAGAS-->>Eval: Metric scores
    Eval->>Report: Write result and scorecard
```

## RAGAS Dataset Mapping

Before calling RAGAS, raw answer records are converted into the RAGAS 0.4.x schema:

```text
question     -> user_input
answer       -> response
contexts     -> retrieved_contexts
ground_truth -> reference
```

## Metric Mapping

```text
RAGAS context_precision  -> retrieval.context_precision
RAGAS context_recall     -> retrieval.context_recall
RAGAS faithfulness       -> generation.faithfulness
RAGAS answer_relevancy   -> generation.response_relevance
RAGAS answer_correctness -> rag_system.completeness
1 - faithfulness         -> rag_system.hallucination_rate
```

## Report Structure

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
    └── run-YYYYMMDDTHHMMSSZ/
        ├── result.json
        ├── result.yaml
        ├── scorecard.json
        └── scorecard.md
```

## LangSmith Dashboard

LangSmith is the visualization layer for the same evaluation artifacts.

```text
RAG trace -> RAGAS feedback scores -> dataset view -> dashboard inspection
```

For setup and dashboard details, see [LANGSMITH_DASHBOARD.md](LANGSMITH_DASHBOARD.md).

## Release Decision Logic

```mermaid
flowchart TD
    A[RAGAS scores] --> B[Average scores across 5 questions]
    B --> C[Compare against manifest gates]
    C --> D{All gates pass?}
    D -->|Yes| E[release_decision.approved = true]
    D -->|No| F[release_decision.approved = false]
    F --> G[List blocking_issues]
```

Quality gates are read from:

```text
manifests/employee-rag-v1.0.0.yaml
```

Current gate rules:

```text
context_relevance >= min_context_relevance
faithfulness >= min_faithfulness
response_relevance >= min_response_relevance
completeness >= min_completeness
hallucination_rate <= max_hallucination_rate
```
