# Simple RAGAS Evaluation Flow

This is the high-level flow of the project without LangSmith.

## One-Page Flow

```mermaid
flowchart TD
    A[Employee Handbook<br/>data/employee_handbook.txt] --> B[Chunk Text<br/>chunk_size=800<br/>chunk_overlap=150]
    B --> C[Create Embeddings<br/>Voyage AI voyage-3-large]
    C --> D[Store Vectors<br/>ChromaDB]

    E[Evaluation Dataset<br/>evaluation/qa_dataset.json<br/>5 questions] --> F[Generate Answers<br/>evaluation/generate_answers.py]
    D --> F
    F --> G[Retrieve Top-K Contexts<br/>top_k=5]
    G --> H[Generate Answer<br/>NVIDIA Llama 3.3 70B Instruct]
    H --> I[Raw RAG Answers<br/>reports/raw_rag_answers/Q001-Q005.json]

    I --> J[RAGAS Evaluation<br/>evaluation/ragas_eval.py]
    J --> K[RAGAS Metrics]
    K --> L[Per-Question Report<br/>result.json / result.yaml]
    K --> M[Release Scorecard<br/>scorecard.json / scorecard.md]

    N[Manifest<br/>manifests/employee-rag-v1.0.0.yaml] --> F
    N --> J
    N --> M
```

## What Each Step Does

```text
1. Ingest handbook
   Converts employee_handbook.txt into chunks and stores embeddings in ChromaDB.

2. Generate raw RAG answers
   Runs each dataset question through the real RAG system.

3. Save raw answer files
   Stores question, generated answer, retrieved contexts, ground truth, latency, and tokens.

4. Run RAGAS
   Evaluates question + answer + contexts + ground truth using real RAGAS metrics.

5. Generate reports
   Writes per-question result files and an overall scorecard.
```

## Command Flow

```mermaid
flowchart LR
    A[python src/ingest.py] --> B[python evaluation/generate_answers.py --force]
    B --> C[python evaluation/ragas_eval.py]
    C --> D[reports/raga_eval/run_id/result.yaml]
    C --> E[reports/raga_eval/run_id/scorecard.json]
```

## RAGAS Metric Mapping

```text
context_precision  -> retrieval.context_precision
context_recall     -> retrieval.context_recall
faithfulness       -> generation.faithfulness
answer_relevancy   -> generation.response_relevance
answer_correctness -> rag_system.completeness
1 - faithfulness   -> rag_system.hallucination_rate
```

## Final Output

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
