import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.config import (  # noqa: E402
    ANSWER_RECORDS_PATH,
    QA_DATASET_PATH,
    REPORT_RESULT_JSON_PATH,
    REPORT_RESULT_YAML_PATH,
    SCORECARD_MD_PATH,
)
from src.manifest import evaluation_metadata, load_manifest, quality_gates  # noqa: E402
from evaluation.metrics import build_report, write_report  # noqa: E402

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "must",
    "of",
    "or",
    "per",
    "should",
    "the",
    "to",
    "what",
    "when",
    "with",
}


def tokenize(text: str) -> set[str]:
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return {token for token in cleaned.split() if token and token not in STOPWORDS}


def coverage(candidate: str, reference: str) -> float:
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    if not reference_tokens:
        return 0.0
    return len(candidate_tokens & reference_tokens) / len(reference_tokens)


def f1(candidate: str, reference: str) -> float:
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0

    overlap = len(candidate_tokens & reference_tokens)
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def context_precision_score(record: dict[str, object]) -> float:
    contexts = [str(context) for context in record["contexts"]]
    signal = tokenize(f"{record['question']} {record['ground_truth']}")
    if not contexts or not signal:
        return 0.0

    relevant_contexts = 0
    for context in contexts:
        context_tokens = tokenize(context)
        if len(context_tokens & signal) >= 2:
            relevant_contexts += 1

    return relevant_contexts / len(contexts)


def gate_status(passed: bool) -> str:
    return "pass" if passed else "fail"


def score_record(record: dict[str, object]) -> dict[str, object]:
    manifest = load_manifest()
    metadata = evaluation_metadata(manifest)
    gates = quality_gates(manifest)
    context = "\n\n".join(str(chunk) for chunk in record["contexts"])
    answer = str(record["answer"])
    ground_truth = str(record["ground_truth"])

    faithfulness_score = coverage(context, answer)
    correctness_score = f1(answer, ground_truth)
    answer_relevancy_score = max(correctness_score, f1(answer, str(record["question"])))
    context_precision = context_precision_score(record)
    context_recall = coverage(context, ground_truth)

    return {
        "release_id": metadata["release_id"],
        "id": record["id"],
        "question": record["question"],
        "answer": answer,
        "ground_truth": ground_truth,
        "faithfulness": round(faithfulness_score, 4),
        "faithfulness_status": gate_status(faithfulness_score >= gates["faithfulness"]),
        "answer_relevancy": round(answer_relevancy_score, 4),
        "answer_relevancy_status": gate_status(
            answer_relevancy_score >= gates["answer_relevancy"]
        ),
        "answer_correctness": round(correctness_score, 4),
        "answer_correctness_status": gate_status(
            correctness_score >= gates["answer_correctness"]
        ),
        "context_precision": round(context_precision, 4),
        "context_precision_status": gate_status(context_precision >= gates["context_precision"]),
        "context_recall": round(context_recall, 4),
        "context_recall_status": gate_status(context_recall >= gates["context_recall"]),
    }


def build_run_id() -> str:
    return f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def load_dataset_ids() -> list[str]:
    rows = json.loads(QA_DATASET_PATH.read_text(encoding="utf-8"))
    return [row["id"] for row in rows]


def build_offline_results() -> list[dict[str, object]]:
    if not ANSWER_RECORDS_PATH.exists():
        raise FileNotFoundError(
            f"Answer records not found: {ANSWER_RECORDS_PATH}. "
            "Run evaluation/generate_answers.py first to create RAG outputs."
        )

    dataset_ids = load_dataset_ids()
    records = json.loads(ANSWER_RECORDS_PATH.read_text(encoding="utf-8"))
    record_by_id = {record["id"]: record for record in records if record["id"] in dataset_ids}
    missing_ids = [dataset_id for dataset_id in dataset_ids if dataset_id not in record_by_id]
    if missing_ids:
        raise ValueError(
            f"Missing answer records for dataset IDs: {', '.join(missing_ids)}. "
            "Run evaluation/generate_answers.py first."
        )
    return [score_record(record_by_id[dataset_id]) for dataset_id in dataset_ids]


def main() -> None:
    run_id = build_run_id()
    results = build_offline_results()
    for result in results:
        result["run_id"] = run_id

    report = build_report(run_id=run_id, results=results)
    write_report(report)
    results_df = pd.DataFrame(results)
    print(results_df)
    print(f"Saved evaluation report to {REPORT_RESULT_JSON_PATH}")
    print(f"Saved evaluation report to {REPORT_RESULT_YAML_PATH}")
    print(f"Saved scorecard to {SCORECARD_MD_PATH}")


if __name__ == "__main__":
    main()
