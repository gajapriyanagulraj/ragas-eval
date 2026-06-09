# Simple RAG + RAGAS Evaluation

This project is a learning-focused Retrieval-Augmented Generation application for an employee handbook.

## Architecture

```text
manifests/employee-rag-v1.0.0.yaml
     |
employee_handbook.txt
     |
Document Loader
     |
Chunking
     |
Voyage AI Embeddings
     |
ChromaDB
     |
Retriever
     |
NVIDIA Llama 3.3 70B Instruct
     |
Answer + Sources
     |
RAGAS Evaluation + Scorecard
```

For the full architecture mindmap and end-to-end flow, see [ARCHITECTURE_FLOW.md](ARCHITECTURE_FLOW.md).
For a simple one-page RAGAS flow diagram, see [SIMPLE_FLOW_DIAGRAM.md](SIMPLE_FLOW_DIAGRAM.md).
For LangSmith observability setup and dashboard details, see [LANGSMITH_DASHBOARD.md](LANGSMITH_DASHBOARD.md).

## Setup

1. Create and activate a Python environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Edit `.env` with real API keys.

```env
VOYAGE_API_KEY=your_voyage_api_key
NVIDIA_API_KEY=your_nvidia_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=employee-rag-evaluation
```

4. Ingest the handbook into ChromaDB.

```bash
python src/ingest.py
```

Expected output:

```text
All chunks indexed in ChromaDB: <number> chunks
```

5. Run the Streamlit UI.

```bash
streamlit run ui/app.py
```

6. Run RAGAS evaluation using the manifest metadata.

```bash
python evaluation/ragas_eval.py
```

This scoring step uses real RAGAS metrics from saved answer records. It uses NVIDIA Llama 3.3 70B Instruct as the evaluator LLM and Voyage as evaluator embeddings. The evaluation pipeline writes JSON/YAML results with a `run_id`:

```text
evaluation/results/answer_records.json
reports/raw_rag_answers/Q001.json
reports/raw_rag_answers/Q002.json
reports/raw_rag_answers/Q003.json
reports/raw_rag_answers/Q004.json
reports/raw_rag_answers/Q005.json
reports/raga_eval/<run_id>/result.json
reports/raga_eval/<run_id>/result.yaml
reports/raga_eval/<run_id>/scorecard.md
reports/raga_eval/<run_id>/scorecard.json
```

You can also run each stage separately:

```bash
python evaluation/generate_answers.py
python evaluation/ragas_eval.py
```

## Sample Questions

- How many annual leave days are employees entitled to?
- Can employees work remotely from another country?
- What is the reimbursement limit for travel expenses?
- How often must passwords be changed?
- How frequently are performance reviews conducted?
- Can confidential files be stored on personal cloud drives?

## Project Layout

```text
data/employee_handbook.txt
chroma_db/
src/config.py
src/loader.py
src/chunker.py
src/embedder.py
src/vectordb.py
src/retriever.py
src/llm.py
src/rag.py
src/ingest.py
src/manifest.py
evaluation/qa_dataset.json
evaluation/generate_answers.py
evaluation/ragas_eval.py
evaluation/metrics.py
evaluation/results/
manifests/employee-rag-v1.0.0.yaml
reports/
ui/app.py
.env
requirements.txt
README.md
```

## Manifest

The active evaluation manifest is:

```text
manifests/employee-rag-v1.0.0.yaml
```

It freezes the release ID, LLM, embedding model, chunk size, overlap, `top_k`, collection name, dataset version, metrics, latency/token capture settings, and quality targets for a reproducible evaluation run.
