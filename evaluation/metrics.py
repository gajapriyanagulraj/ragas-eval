import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.config import (  # noqa: E402
    ANSWER_RECORDS_PATH,
    RAGA_EVAL_DIR,
)
from src.manifest import evaluation_metadata, load_manifest, quality_gates  # noqa: E402


def _mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(numeric.mean()), 4) if not numeric.empty else None


def _mean_required(series: pd.Series) -> float:
    value = _mean(series)
    return value if value is not None else 0.0


def _score(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


def _derived_hallucination_rate(faithfulness_score: float | None) -> float | None:
    if faithfulness_score is None:
        return None
    return round(1 - faithfulness_score, 4)


def _metric_status(score: float | None, gate: float, *, lower_is_better: bool = False) -> str:
    if score is None:
        return "NOT_EVALUATED"
    if lower_is_better:
        return gate_status(score <= gate)
    return gate_status(score >= gate)


def display_value(value: object) -> object:
    return "null" if value is None else value


def _load_answer_records() -> pd.DataFrame:
    if not ANSWER_RECORDS_PATH.exists():
        raise FileNotFoundError(f"Answer records not found: {ANSWER_RECORDS_PATH}")
    return pd.DataFrame(json.loads(ANSWER_RECORDS_PATH.read_text(encoding="utf-8")))


def _answer_records_by_id(answers_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {
        str(record["id"]): record
        for record in answers_df.to_dict(orient="records")
        if "id" in record
    }


def gate_status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def gate_reason(metric: str, score: float | None, gate: float, status: str) -> str:
    if score is None:
        return f"{metric} was not successfully computed by RAGAS."

    if status == "PASS":
        if metric == "hallucination_rate":
            return f"{score} is at or below the allowed maximum of {gate}."
        return f"{score} meets or exceeds the threshold of {gate}."

    if metric == "hallucination_rate":
        return f"{score} is above the allowed maximum of {gate}."
    return f"{score} is below the required threshold of {gate}."


def build_gate_report(
    scores: dict[str, float | None],
    gates: dict[str, float],
    quality_gate: dict[str, str],
) -> dict[str, dict[str, object]]:
    return {
        "context_relevance": {
            "actual": scores["context_precision"],
            "target": gates["context_precision"],
            "status": quality_gate["context_relevance"],
            "rule": "actual >= target",
        },
        "faithfulness": {
            "actual": scores["faithfulness"],
            "target": gates["faithfulness"],
            "status": quality_gate["faithfulness"],
            "rule": "actual >= target",
        },
        "response_relevance": {
            "actual": scores["response_relevance"],
            "target": gates["answer_relevancy"],
            "status": quality_gate["response_relevance"],
            "rule": "actual >= target",
        },
        "completeness": {
            "actual": scores["completeness"],
            "target": gates["answer_correctness"],
            "status": quality_gate["completeness"],
            "rule": "actual >= target",
        },
        "hallucination_rate": {
            "actual": scores["hallucination_rate"],
            "target": gates["hallucination_rate"],
            "status": quality_gate["hallucination_rate"],
            "rule": "actual <= target",
        },
    }


def build_question_reports(
    results: list[dict[str, object]],
    answers_df: pd.DataFrame,
    gates: dict[str, float],
) -> list[dict[str, object]]:
    answer_records = _answer_records_by_id(answers_df)
    question_reports = []

    for result in results:
        question_id = str(result["id"])
        answer_record = answer_records.get(question_id, {})
        faithfulness_score = _score(result["faithfulness"])
        answer_relevancy_score = _score(result["answer_relevancy"])
        answer_correctness_score = _score(result["answer_correctness"])
        context_precision_score = _score(result["context_precision"])
        context_recall_score = _score(result["context_recall"])
        hallucination_rate = _derived_hallucination_rate(faithfulness_score)
        question_statuses = [
            _metric_status(context_precision_score, gates["context_precision"]),
            _metric_status(context_recall_score, gates["context_recall"]),
            _metric_status(faithfulness_score, gates["faithfulness"]),
            _metric_status(answer_correctness_score, gates["answer_correctness"]),
            _metric_status(hallucination_rate, gates["hallucination_rate"], lower_is_better=True),
        ]
        question_status = "PASS" if all(status == "PASS" for status in question_statuses) else "FAIL"

        question_reports.append(
            {
                "id": question_id,
                "question": result["question"],
                "expected_answer": result["ground_truth"],
                "generated_answer": result["answer"],
                "retrieval": {
                    "context_precision": context_precision_score,
                    "context_recall": context_recall_score,
                },
                "generation": {
                    "faithfulness": faithfulness_score,
                    "response_relevance": answer_relevancy_score,
                },
                "rag_system": {
                    "completeness": answer_correctness_score,
                    "hallucination_rate": hallucination_rate,
                },
                "performance": {
                    "latency_ms": answer_record.get("total_latency_ms", 0),
                    "prompt_tokens": answer_record.get("input_tokens", 0),
                    "completion_tokens": answer_record.get("output_tokens", 0),
                    "total_tokens": answer_record.get("total_tokens", 0),
                },
                "status": question_status,
            }
        )

    return question_reports


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
    hallucination_rate = _derived_hallucination_rate(faithfulness_score)
    metric_columns = [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_precision",
        "context_recall",
    ]

    scores = {
        "context_precision": context_precision_score,
        "context_recall": context_recall_score,
        "context_relevance": context_precision_score,
        "faithfulness": faithfulness_score,
        "response_relevance": answer_relevancy_score,
        "completeness": answer_correctness_score,
        "hallucination_rate": hallucination_rate,
    }
    quality_gate = {
        "faithfulness": _metric_status(faithfulness_score, gates["faithfulness"]),
        "response_relevance": _metric_status(answer_relevancy_score, gates["answer_relevancy"]),
        "completeness": _metric_status(answer_correctness_score, gates["answer_correctness"]),
        "context_relevance": _metric_status(context_precision_score, gates["context_precision"]),
        "hallucination_rate": _metric_status(
            hallucination_rate,
            gates["hallucination_rate"],
            lower_is_better=True,
        ),
    }
    gate_details = {
        "faithfulness": {
            "score": scores["faithfulness"],
            "threshold": gates["faithfulness"],
            "status": quality_gate["faithfulness"],
            "calculation": "RAGAS faithfulness: semantic support of the answer by retrieved contexts.",
            "reason": gate_reason(
                "faithfulness",
                scores["faithfulness"],
                gates["faithfulness"],
                quality_gate["faithfulness"],
            ),
        },
        "response_relevance": {
            "score": scores["response_relevance"],
            "threshold": gates["answer_relevancy"],
            "status": quality_gate["response_relevance"],
            "calculation": "RAGAS answer_relevancy: semantic relevance of answer to question.",
            "reason": gate_reason(
                "response_relevance",
                scores["response_relevance"],
                gates["answer_relevancy"],
                quality_gate["response_relevance"],
            ),
        },
        "completeness": {
            "score": scores["completeness"],
            "threshold": gates["answer_correctness"],
            "status": quality_gate["completeness"],
            "calculation": "RAGAS answer_correctness: semantic match between answer and ground truth.",
            "reason": gate_reason(
                "completeness",
                scores["completeness"],
                gates["answer_correctness"],
                quality_gate["completeness"],
            ),
        },
        "context_relevance": {
            "score": scores["context_precision"],
            "threshold": gates["context_precision"],
            "status": quality_gate["context_relevance"],
            "calculation": "RAGAS context_precision: usefulness of retrieved contexts for the answer.",
            "reason": gate_reason(
                "context_relevance",
                scores["context_precision"],
                gates["context_precision"],
                quality_gate["context_relevance"],
            ),
        },
        "hallucination_rate": {
            "score": scores["hallucination_rate"],
            "threshold": gates["hallucination_rate"],
            "status": quality_gate["hallucination_rate"],
            "calculation": "Derived metric: 1 - faithfulness.",
            "reason": gate_reason(
                "hallucination_rate",
                scores["hallucination_rate"],
                gates["hallucination_rate"],
                quality_gate["hallucination_rate"],
            ),
        },
    }
    metric_coverage = {
        metric: {
            "scored": int(results_df[metric].notna().sum()),
            "total": int(len(results_df)),
        }
        for metric in metric_columns
    }
    for coverage in metric_coverage.values():
        if coverage["scored"] == coverage["total"]:
            coverage["status"] = "COMPLETE"
        elif coverage["scored"] == 0:
            coverage["status"] = "FAILED"
        else:
            coverage["status"] = "PARTIAL"

    metrics_requested = len(metric_columns)
    metrics_complete = sum(
        1 for coverage in metric_coverage.values() if coverage["status"] == "COMPLETE"
    )
    metrics_scored = sum(
        1 for coverage in metric_coverage.values() if coverage["scored"] > 0
    )
    metrics_failed = sum(
        1 for coverage in metric_coverage.values() if coverage["status"] == "FAILED"
    )
    metrics_partial = sum(
        1 for coverage in metric_coverage.values() if coverage["status"] == "PARTIAL"
    )
    gates_passed = sum(1 for status in quality_gate.values() if status == "PASS")
    gates_failed = sum(1 for status in quality_gate.values() if status == "FAIL")
    gates_not_evaluated = sum(
        1 for status in quality_gate.values() if status == "NOT_EVALUATED"
    )
    overall_status = "INCOMPLETE"
    if metrics_complete == metrics_requested:
        overall_status = "PASS" if gates_failed == 0 else "FAIL"

    return {
        "run_id": run_id,
        "release_id": metadata["release_id"],
        "manifest_file": "manifests/employee-rag-v1.0.0.yaml",
        "evaluator": metadata["evaluator"],
        "summary": {
            "questions_evaluated": int(len(results_df)),
            "metrics_requested": metrics_requested,
            "metrics_complete": metrics_complete,
            "metrics_scored": metrics_scored,
            "metrics_partial": metrics_partial,
            "metrics_failed": metrics_failed,
            "gates_passed": gates_passed,
            "gates_failed": gates_failed,
            "gates_not_evaluated": gates_not_evaluated,
            "overall_status": overall_status,
        },
        "gates": build_gate_report(scores, gates, quality_gate),
        "threshold_validation": {
            "status": "PASS",
            "rule": "all min/max gates are required numeric values between 0 and 1",
        },
        "scoring_method": "RAGAS LLM-as-judge metrics",
        "retrieval": {
            "context_precision": scores["context_precision"],
            "context_recall": scores["context_recall"],
            "context_relevance": scores["context_relevance"],
        },
        "generation": {
            "faithfulness": scores["faithfulness"],
            "response_relevance": scores["response_relevance"],
        },
        "rag_system": {
            "completeness": scores["completeness"],
            "hallucination_rate": scores["hallucination_rate"],
        },
        "scores": scores,
        "metric_coverage": metric_coverage,
        "quality_gate": quality_gate,
        "gate_details": gate_details,
        "performance": {
            "avg_retrieval_latency_ms": _mean_required(answers_df["retrieval_time_ms"]),
            "avg_generation_latency_ms": _mean_required(answers_df["generation_time_ms"]),
            "avg_total_latency_ms": _mean_required(answers_df["total_latency_ms"]),
        },
        "tokens": {
            "avg_prompt_tokens": _mean_required(answers_df["input_tokens"]),
            "avg_completion_tokens": _mean_required(answers_df["output_tokens"]),
            "avg_total_tokens": _mean_required(answers_df["total_tokens"]),
        },
        "question_results": build_question_reports(results, answers_df, gates),
    }


def write_report(
    report: dict[str, object],
    output_dir: Path | None = None,
) -> dict[str, Path]:
    target_dir = output_dir or RAGA_EVAL_DIR / str(report["run_id"])
    result_json_path = target_dir / "result.json"
    result_yaml_path = target_dir / "result.yaml"
    scorecard_json_path = target_dir / "scorecard.json"
    scorecard_md_path = target_dir / "scorecard.md"

    target_dir.mkdir(parents=True, exist_ok=True)
    result_payload = build_result_payload(report)
    scorecard = build_scorecard_payload(report)

    result_json_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    result_yaml_path.write_text(
        yaml.safe_dump(result_payload, sort_keys=False),
        encoding="utf-8",
    )
    scorecard_json_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")

    summary = report["summary"]
    retrieval = report["retrieval"]
    generation = report["generation"]
    rag_system = report["rag_system"]
    coverage = report["metric_coverage"]
    performance = report["performance"]
    tokens = report["tokens"]
    gates = report["gates"]
    overall_status = summary["overall_status"]

    markdown = f"""# Employee RAG Evaluation Report

Run ID: {report["run_id"]}

Release: {report["release_id"]}

Manifest: `{report["manifest_file"]}`

Threshold validation: {report["threshold_validation"]["status"]}

Scoring method: {report["scoring_method"]}

## Configuration

Model:
- {load_manifest()["model"]["name"]}

Embeddings:
- {load_manifest()["retrieval"]["embedding_model"]}

Vector Store:
- {load_manifest()["retrieval"]["vector_store"]}

Dataset:
- {summary["questions_evaluated"]} evaluation questions

## Retrieval Evaluation

| Metric | Score |
| --- | ---: |
| Context Precision | {display_value(retrieval["context_precision"])} |
| Context Recall | {display_value(retrieval["context_recall"])} |
| Context Relevance | {display_value(retrieval["context_relevance"])} |

## Generation Evaluation

| Metric | Score |
| --- | ---: |
| Faithfulness | {display_value(generation["faithfulness"])} |
| Response Relevance | {display_value(generation["response_relevance"])} |

## RAG System Evaluation

| Metric | Score |
| --- | ---: |
| Completeness | {display_value(rag_system["completeness"])} |
| Hallucination Rate | {display_value(rag_system["hallucination_rate"])} |

## Performance

| Metric | Value |
| --- | ---: |
| Avg Retrieval Latency | {performance["avg_retrieval_latency_ms"]} ms |
| Avg Generation Latency | {performance["avg_generation_latency_ms"]} ms |
| Avg Total Latency | {performance["avg_total_latency_ms"]} ms |
| Avg Prompt Tokens | {tokens["avg_prompt_tokens"]} |
| Avg Completion Tokens | {tokens["avg_completion_tokens"]} |
| Avg Total Tokens | {tokens["avg_total_tokens"]} |

## Gate Validation

Overall status: {overall_status}

Metric coverage: {summary["metrics_complete"]} complete, {summary["metrics_partial"]} partial, {summary["metrics_failed"]} failed

Gate result: {summary["gates_passed"]} passed, {summary["gates_failed"]} failed, {summary["gates_not_evaluated"]} not evaluated

| Gate | Actual | Target | Rule | Status |
| --- | ---: | ---: | --- | --- |
| Context Relevance | {display_value(gates["context_relevance"]["actual"])} | {gates["context_relevance"]["target"]} | {gates["context_relevance"]["rule"]} | {gates["context_relevance"]["status"]} |
| Faithfulness | {display_value(gates["faithfulness"]["actual"])} | {gates["faithfulness"]["target"]} | {gates["faithfulness"]["rule"]} | {gates["faithfulness"]["status"]} |
| Response Relevance | {display_value(gates["response_relevance"]["actual"])} | {gates["response_relevance"]["target"]} | {gates["response_relevance"]["rule"]} | {gates["response_relevance"]["status"]} |
| Completeness | {display_value(gates["completeness"]["actual"])} | {gates["completeness"]["target"]} | {gates["completeness"]["rule"]} | {gates["completeness"]["status"]} |
| Hallucination Rate | {display_value(gates["hallucination_rate"]["actual"])} | {gates["hallucination_rate"]["target"]} | {gates["hallucination_rate"]["rule"]} | {gates["hallucination_rate"]["status"]} |

## Why Gates Failed

{format_failure_reasons(report)}

## Metric Coverage

| Metric | Scored | Total | Status |
| --- | ---: | ---: | --- |
| Faithfulness | {coverage["faithfulness"]["scored"]} | {coverage["faithfulness"]["total"]} | {coverage["faithfulness"]["status"]} |
| Response Relevance | {coverage["answer_relevancy"]["scored"]} | {coverage["answer_relevancy"]["total"]} | {coverage["answer_relevancy"]["status"]} |
| Completeness | {coverage["answer_correctness"]["scored"]} | {coverage["answer_correctness"]["total"]} | {coverage["answer_correctness"]["status"]} |
| Context Precision | {coverage["context_precision"]["scored"]} | {coverage["context_precision"]["total"]} | {coverage["context_precision"]["status"]} |
| Context Recall | {coverage["context_recall"]["scored"]} | {coverage["context_recall"]["total"]} | {coverage["context_recall"]["status"]} |
"""
    scorecard_md_path.write_text(markdown, encoding="utf-8")
    return {
        "result_json": result_json_path,
        "result_yaml": result_yaml_path,
        "scorecard_json": scorecard_json_path,
        "scorecard_md": scorecard_md_path,
    }


def build_result_payload(report: dict[str, object]) -> dict[str, object]:
    return {
        "release_id": report["release_id"],
        "run_id": report["run_id"],
        "questions": report["question_results"],
    }


def build_scorecard_payload(report: dict[str, object]) -> dict[str, object]:
    gates = report["gates"]
    failed_gates = [
        gate_name
        for gate_name, gate in gates.items()
        if gate["status"] in {"FAIL", "NOT_EVALUATED"}
    ]

    return {
        "release_id": report["release_id"],
        "run_id": report["run_id"],
        "status": report["summary"]["overall_status"],
        "summary": {
            "questions_evaluated": report["summary"]["questions_evaluated"],
            "gates_passed": report["summary"]["gates_passed"],
            "gates_failed": report["summary"]["gates_failed"],
        },
        "metrics": {
            "retrieval": report["retrieval"],
            "generation": report["generation"],
            "rag_system": report["rag_system"],
        },
        "performance": {
            **report["performance"],
            **report["tokens"],
        },
        "quality_gates": {
            gate_name: {
                "actual": gate["actual"],
                "threshold": gate["target"],
                "status": gate["status"],
            }
            for gate_name, gate in gates.items()
        },
        "release_decision": {
            "approved": report["summary"]["overall_status"] == "PASS",
            "blocking_issues": failed_gates,
        },
    }


def build_evaluation_status(overall_status: str) -> dict[str, str]:
    if overall_status == "INCOMPLETE":
        return {
            "status": "INCOMPLETE",
            "recommendation": (
                "Do not use this report for release approval. RAGAS evaluation "
                "coverage is incomplete and required metrics were not successfully scored."
            ),
        }
    if overall_status == "FAIL":
        return {
            "status": "FAIL",
            "recommendation": (
                "Do not approve this release until failed quality gates are remediated."
            ),
        }
    return {
        "status": "PASS",
        "recommendation": "All required RAGAS metrics were scored and release gates passed.",
    }


def build_failure_reasons(report: dict[str, object]) -> list[dict[str, str]]:
    reasons = []
    metric_actions = {
        "faithfulness": "Verify evaluator LLM, retrieved contexts, and answer format.",
        "answer_relevancy": "Review answer generation and evaluator configuration.",
        "answer_correctness": "Verify ground_truth field and evaluator configuration.",
        "context_precision": "Verify contexts format and retrieval dataset.",
        "context_recall": "Verify contexts and ground_truth fields.",
    }

    for metric, coverage in report["metric_coverage"].items():
        if coverage["status"] == "FAILED":
            reasons.append(
                {
                    "metric": metric,
                    "reason": "Metric was not successfully computed by RAGAS.",
                    "action": metric_actions[metric],
                }
            )
        elif coverage["status"] == "PARTIAL":
            reasons.append(
                {
                    "metric": metric,
                    "reason": (
                        f"Metric was computed for {coverage['scored']} of "
                        f"{coverage['total']} questions."
                    ),
                    "action": metric_actions[metric],
                }
            )

    for metric, detail in report["gate_details"].items():
        if detail["status"] == "FAIL":
            reasons.append(
                {
                    "metric": metric,
                    "reason": detail["reason"],
                    "action": "Review RAG retrieval, generation prompt, and evaluator configuration.",
                }
            )

    return reasons


def format_failure_reasons(report: dict[str, object]) -> str:
    reasons = build_failure_reasons(report)
    if not reasons:
        return "All required metrics were scored and all quality gates passed."

    return "\n".join(
        f"- `{reason['metric']}`: {reason['reason']} Action: {reason['action']}"
        for reason in reasons
    )


def main() -> None:
    raise SystemExit("Run `python evaluation/ragas_eval.py` to generate the report.")


if __name__ == "__main__":
    main()
