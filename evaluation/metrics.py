import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.config import (  # noqa: E402
    ANSWER_RECORDS_PATH,
    REPORT_RESULT_JSON_PATH,
    REPORT_RESULT_YAML_PATH,
    SCORECARD_JSON_PATH,
    SCORECARD_MD_PATH,
)
from src.manifest import evaluation_metadata, load_manifest, quality_gates  # noqa: E402


def _mean(series: pd.Series) -> float:
    return round(float(series.mean()), 4) if not series.empty else 0.0


def _load_answer_records() -> pd.DataFrame:
    if not ANSWER_RECORDS_PATH.exists():
        raise FileNotFoundError(f"Answer records not found: {ANSWER_RECORDS_PATH}")
    return pd.DataFrame(json.loads(ANSWER_RECORDS_PATH.read_text(encoding="utf-8")))


def gate_status(passed: bool) -> str:
    return "pass" if passed else "fail"


def build_report(run_id: str, results: list[dict[str, object]]) -> dict[str, object]:
    answers_df = _load_answer_records()
    results_df = pd.DataFrame(results)
    manifest = load_manifest()
    metadata = evaluation_metadata(manifest)
    gates = quality_gates(manifest)

    faithfulness_score = _mean(results_df["faithfulness"])
    answer_relevancy_score = _mean(results_df["answer_relevancy"])
    answer_correctness_score = _mean(results_df["answer_correctness"])
    context_precision_score = _mean(results_df["context_precision"])
    context_recall_score = _mean(results_df["context_recall"])
    hallucination_rate = round(1 - faithfulness_score, 4)

    return {
        "run_id": run_id,
        "release_id": metadata["release_id"],
        "manifest_file": "manifests/employee-rag-v1.0.0.yaml",
        "gates": manifest["gates"],
        "threshold_validation": {
            "status": "pass",
            "rule": "all gates are required numeric values between 0 and 1",
        },
        "scoring_method": "offline lexical proxy metrics; no Groq or Voyage judge/scoring calls",
        "scores": {
            "context_precision": context_precision_score,
            "context_recall": context_recall_score,
            "faithfulness": faithfulness_score,
            "response_relevance": answer_relevancy_score,
            "completeness": answer_correctness_score,
            "hallucination_rate": hallucination_rate,
        },
        "quality_gate": {
            "faithfulness": gate_status(faithfulness_score >= gates["faithfulness"]),
            "response_relevance": gate_status(
                answer_relevancy_score >= gates["answer_relevancy"]
            ),
            "completeness": gate_status(answer_correctness_score >= gates["answer_correctness"]),
            "context_relevance": gate_status(context_precision_score >= gates["context_precision"]),
            "context_recall": gate_status(context_recall_score >= gates["context_recall"]),
            "hallucination_rate": gate_status(hallucination_rate <= gates["hallucination_rate"]),
        },
        "performance": {
            "avg_retrieval_latency_ms": _mean(answers_df["retrieval_time_ms"]),
            "avg_generation_latency_ms": _mean(answers_df["generation_time_ms"]),
            "avg_total_latency_ms": _mean(answers_df["total_latency_ms"]),
        },
        "tokens": {
            "avg_prompt_tokens": _mean(answers_df["input_tokens"]),
            "avg_completion_tokens": _mean(answers_df["output_tokens"]),
            "avg_total_tokens": _mean(answers_df["total_tokens"]),
        },
        "question_results": results,
    }


def write_report(report: dict[str, object]) -> None:
    REPORT_RESULT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_RESULT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_RESULT_YAML_PATH.write_text(
        yaml.safe_dump(report, sort_keys=False),
        encoding="utf-8",
    )
    SCORECARD_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    scores = report["scores"]
    performance = report["performance"]
    tokens = report["tokens"]
    quality_gate = report["quality_gate"]

    markdown = f"""# RAG Evaluation Report

Run ID: {report["run_id"]}

Release: {report["release_id"]}

Manifest: `{report["manifest_file"]}`

Threshold validation: {report["threshold_validation"]["status"]}

Scoring method: {report["scoring_method"]}

## Scores

| Metric | Score | Gate | Status |
| --- | ---: | ---: | --- |
| Faithfulness | {scores["faithfulness"]} | {report["gates"]["faithfulness"]} | {quality_gate["faithfulness"]} |
| Response Relevance | {scores["response_relevance"]} | {report["gates"]["response_relevance"]} | {quality_gate["response_relevance"]} |
| Completeness | {scores["completeness"]} | {report["gates"]["completeness"]} | {quality_gate["completeness"]} |
| Context Relevance | {scores["context_precision"]} | {report["gates"]["context_relevance"]} | {quality_gate["context_relevance"]} |
| Context Recall | {scores["context_recall"]} | {report["gates"]["context_relevance"]} | {quality_gate["context_recall"]} |
| Hallucination Rate | {scores["hallucination_rate"]} | {report["gates"]["hallucination_rate"]} | {quality_gate["hallucination_rate"]} |

## Performance

| Metric | Value |
| --- | ---: |
| Avg Retrieval Latency | {performance["avg_retrieval_latency_ms"]} ms |
| Avg Generation Latency | {performance["avg_generation_latency_ms"]} ms |
| Avg Total Latency | {performance["avg_total_latency_ms"]} ms |
| Avg Prompt Tokens | {tokens["avg_prompt_tokens"]} |
| Avg Completion Tokens | {tokens["avg_completion_tokens"]} |
| Avg Total Tokens | {tokens["avg_total_tokens"]} |
"""
    SCORECARD_MD_PATH.write_text(markdown, encoding="utf-8")


def main() -> None:
    raise SystemExit("Run `python evaluation/ragas_eval.py` to generate the report.")


if __name__ == "__main__":
    main()
