from pathlib import Path

from dotenv import load_dotenv
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
EVALUATION_DIR = ROOT_DIR / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"
MANIFESTS_DIR = ROOT_DIR / "manifests"
REPORTS_DIR = ROOT_DIR / "reports"

HANDBOOK_PATH = DATA_DIR / "employee_handbook.txt"
QA_DATASET_PATH = EVALUATION_DIR / "qa_dataset.json"
ANSWER_RECORDS_PATH = RESULTS_DIR / "answer_records.json"
RAGAS_RESULTS_JSON_PATH = RESULTS_DIR / "ragas_results.json"
RAGAS_RESULTS_YAML_PATH = RESULTS_DIR / "ragas_results.yaml"
REPORTS_RESULTS_JSON_PATH = REPORTS_DIR / "ragas_results.json"
REPORTS_RESULTS_YAML_PATH = REPORTS_DIR / "ragas_results.yaml"
RAGA_EVAL_DIR = REPORTS_DIR / "raga_eval"
REPORT_RESULT_JSON_PATH = RAGA_EVAL_DIR / "result.json"
REPORT_RESULT_YAML_PATH = RAGA_EVAL_DIR / "result.yaml"
SCORECARD_MD_PATH = RAGA_EVAL_DIR / "scorecard.md"
SCORECARD_JSON_PATH = RAGA_EVAL_DIR / "scorecard.json"
RAW_RAG_ANSWERS_DIR = REPORTS_DIR / "raw_rag_answers"
REPORT_RUNS_DIR = REPORTS_DIR / "runs"
REPORT_JSON_PATH = EVALUATION_DIR / "report.json"
REPORT_YAML_PATH = EVALUATION_DIR / "report.yaml"
REPORT_MD_PATH = EVALUATION_DIR / "report.md"
MANIFEST_PATH = MANIFESTS_DIR / "employee-rag-v1.0.0.yaml"

COLLECTION_NAME = "employee_handbook"
VOYAGE_EMBED_MODEL = "voyage-3-large"
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
EMBED_BATCH_SIZE = 8
EMBED_BATCH_DELAY_SECONDS = 22

load_dotenv(ROOT_DIR / ".env")

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
