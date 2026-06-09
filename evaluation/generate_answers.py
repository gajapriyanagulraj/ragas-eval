import json
import sys
import time
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.config import (  # noqa: E402
    ANSWER_RECORDS_PATH,
    EMBED_BATCH_DELAY_SECONDS,
    QA_DATASET_PATH,
    RAW_RAG_ANSWERS_DIR,
)
from src.rag import HandbookRAG  # noqa: E402


def load_qa_dataset() -> list[dict[str, str]]:
    rows = json.loads(QA_DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"QA dataset must be a JSON list: {QA_DATASET_PATH}")
    return rows


def generate_answer_records(force: bool = False) -> list[dict[str, object]]:
    qa_rows = load_qa_dataset()
    rag = HandbookRAG()

    records = []
    if ANSWER_RECORDS_PATH.exists() and not force:
        records = json.loads(ANSWER_RECORDS_PATH.read_text(encoding="utf-8"))

    dataset_ids = {row["id"] for row in qa_rows}
    records = [record for record in records if record["id"] in dataset_ids]
    completed_ids = {record["id"] for record in records}
    for row in qa_rows:
        if row["id"] in completed_ids:
            print(f"Skipping {row['id']}: already generated")
            continue

        print(f"Generating {row['id']}: {row['question']}", flush=True)
        result = rag.ask(row["question"])
        records.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "sources": result["sources"],
                "ground_truth": row["expected_answer"],
                "retrieval_time_ms": result["retrieval_time_ms"],
                "generation_time_ms": result["generation_time_ms"],
                "total_latency_ms": result["total_latency_ms"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "total_tokens": result["total_tokens"],
            }
        )
        save_answer_records(records)
        print(f"Saved {row['id']}", flush=True)

        if len(records) < len(qa_rows):
            print(f"Waiting {EMBED_BATCH_DELAY_SECONDS}s for Voyage rate limit...", flush=True)
            time.sleep(EMBED_BATCH_DELAY_SECONDS)

    record_by_id = {record["id"]: record for record in records}
    return [record_by_id[row["id"]] for row in qa_rows]


def save_answer_records(records: list[dict[str, object]]) -> None:
    ANSWER_RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANSWER_RECORDS_PATH.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    RAW_RAG_ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = RAW_RAG_ANSWERS_DIR / f"{record['id']}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all answer records even if cached records already exist.",
    )
    args = parser.parse_args()

    records = generate_answer_records(force=args.force)
    save_answer_records(records)
    print(f"Saved {len(records)} answer records to {ANSWER_RECORDS_PATH}")
    print(f"Saved raw RAG answer files to {RAW_RAG_ANSWERS_DIR}")


if __name__ == "__main__":
    main()
