import json
import math
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.config import (  # noqa: E402
    ANSWER_RECORDS_PATH,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    HF_BASE_URL,
    HF_EVAL_MODEL,
    HF_TOKEN,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    RAGA_EVAL_DIR,
    VOYAGE_API_KEY,
    VOYAGE_EMBED_MODEL,
)
from src.manifest import evaluation_metadata, load_manifest, quality_gates  # noqa: E402
from evaluation.metrics import build_report, write_report  # noqa: E402
from src.langsmith_observability import (  # noqa: E402
    create_evaluation_summary_run,
    create_feedback_scores,
    create_or_update_dataset,
    ensure_rag_trace,
    flush_langsmith,
)


def build_run_id() -> str:
    return f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def write_error_report(run_id: str, error: Exception) -> dict[str, Path]:
    manifest = load_manifest()
    metadata = evaluation_metadata(manifest)
    target_dir = RAGA_EVAL_DIR / run_id
    target_dir.mkdir(parents=True, exist_ok=True)

    error_payload = {
        "run_id": run_id,
        "release_id": metadata["release_id"],
        "manifest_file": "manifests/employee-rag-v1.0.0.yaml",
        "overall_status": "ERROR",
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "evaluation_status": {
            "status": "ERROR",
            "recommendation": (
                "RAGAS evaluation stopped before scoring completed. Fix the provider "
                "or quota error, then rerun evaluation."
            ),
        },
    }
    scorecard_md = f"""# Employee RAG Evaluation Report

Run ID: {run_id}

Release: {metadata["release_id"]}

Overall status: ERROR

RAGAS evaluation stopped before scoring completed.

Error type: {type(error).__name__}

Error message:

```text
{error}
```

Recommendation:

Fix the provider or quota error, then rerun `python evaluation/ragas_eval.py`.
"""

    error_json_path = target_dir / "run_error.json"
    scorecard_json_path = target_dir / "scorecard.json"
    scorecard_md_path = target_dir / "scorecard.md"
    error_json_path.write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
    scorecard_json_path.write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
    scorecard_md_path.write_text(scorecard_md, encoding="utf-8")
    return {
        "error_json": error_json_path,
        "scorecard_json": scorecard_json_path,
        "scorecard_md": scorecard_md_path,
    }


def load_answer_records() -> list[dict[str, Any]]:
    if not ANSWER_RECORDS_PATH.exists():
        raise FileNotFoundError(
            f"Answer records not found: {ANSWER_RECORDS_PATH}. "
            "Run `python evaluation/generate_answers.py` first."
        )
    records = json.loads(ANSWER_RECORDS_PATH.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Answer records must be a JSON list: {ANSWER_RECORDS_PATH}")
    return records


def build_ragas_dataset(records: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "user_input": record["question"],
                "response": record["answer"],
                "retrieved_contexts": record["contexts"],
                "reference": record["ground_truth"],
            }
            for record in records
        ]
    )


def build_evaluator_llm() -> LangchainLLMWrapper:
    manifest = load_manifest()
    evaluator = manifest["evaluation"].get("evaluator", {})
    provider = evaluator.get("provider", "groq")

    if provider == "huggingface":
        if not HF_TOKEN:
            raise RuntimeError("Set HF_TOKEN in .env before running Hugging Face RAGAS evaluation.")

        return LangchainLLMWrapper(
            ChatOpenAI(
                api_key=HF_TOKEN,
                base_url=evaluator.get("base_url", HF_BASE_URL),
                model=evaluator.get("name", HF_EVAL_MODEL),
                temperature=0,
            )
        )

    if provider == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError("Set GROQ_API_KEY in .env before running RAGAS evaluation.")

        return LangchainLLMWrapper(
            ChatOpenAI(
                api_key=GROQ_API_KEY,
                base_url=evaluator.get("base_url", GROQ_BASE_URL),
                model=evaluator.get("name", GROQ_MODEL),
                temperature=0,
            )
        )

    if provider == "nvidia":
        if not NVIDIA_API_KEY:
            raise RuntimeError("Set NVIDIA_API_KEY in .env before running NVIDIA RAGAS evaluation.")

        return LangchainLLMWrapper(
            ChatOpenAI(
                api_key=NVIDIA_API_KEY,
                base_url=evaluator.get("base_url", NVIDIA_BASE_URL),
                model=evaluator.get("name", NVIDIA_MODEL),
                temperature=0,
            )
        )

    raise RuntimeError(f"Unsupported RAGAS evaluator provider: {provider}")


def build_evaluator_embeddings() -> LangchainEmbeddingsWrapper:
    if not VOYAGE_API_KEY:
        raise RuntimeError("Set VOYAGE_API_KEY in .env before running RAGAS evaluation.")

    return LangchainEmbeddingsWrapper(
        VoyageAIEmbeddings(
            voyage_api_key=VOYAGE_API_KEY,
            model=VOYAGE_EMBED_MODEL,
        )
    )


def build_ragas_metrics() -> list[Any]:
    metrics = [
        deepcopy(faithfulness),
        deepcopy(answer_relevancy),
        deepcopy(answer_correctness),
        deepcopy(context_precision),
        deepcopy(context_recall),
    ]

    for metric in metrics:
        if getattr(metric, "name", "") == "answer_relevancy" and hasattr(metric, "strictness"):
            # Some OpenAI-compatible providers accept one completion per request only.
            metric.strictness = 1

    return metrics


def normalize_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def gate_status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def score_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return round(float(value), 4)


def score_status(score: float | None, gate: float) -> str:
    if score is None:
        return "NOT_SCORED"
    return gate_status(score >= gate)


def add_metadata_and_gate_statuses(
    ragas_df: pd.DataFrame,
    records: list[dict[str, Any]],
    run_id: str,
) -> list[dict[str, Any]]:
    manifest = load_manifest()
    metadata = evaluation_metadata(manifest)
    gates = quality_gates(manifest)

    results: list[dict[str, Any]] = []
    for index, row in ragas_df.iterrows():
        source_record = records[index]
        faithfulness_score = score_value(row.get("faithfulness"))
        answer_relevancy_score = score_value(row.get("answer_relevancy"))
        answer_correctness_score = score_value(row.get("answer_correctness"))
        context_precision_score = score_value(row.get("context_precision"))
        context_recall_score = score_value(row.get("context_recall"))
        langsmith_run_id = ensure_rag_trace(
            question_record=source_record,
            evaluation_run_id=run_id,
        )

        result = {
            "run_id": run_id,
            "release_id": metadata["release_id"],
            "id": source_record["id"],
            "question": source_record["question"],
            "answer": source_record["answer"],
            "ground_truth": source_record["ground_truth"],
            "faithfulness": faithfulness_score,
            "faithfulness_status": score_status(faithfulness_score, gates["faithfulness"]),
            "answer_relevancy": answer_relevancy_score,
            "answer_relevancy_status": score_status(
                answer_relevancy_score,
                gates["answer_relevancy"],
            ),
            "answer_correctness": answer_correctness_score,
            "answer_correctness_status": score_status(
                answer_correctness_score,
                gates["answer_correctness"],
            ),
            "context_precision": context_precision_score,
            "context_precision_status": score_status(
                context_precision_score,
                gates["context_precision"],
            ),
            "context_recall": context_recall_score,
            "context_recall_status": score_status(context_recall_score, gates["context_recall"]),
            "langsmith_run_id": langsmith_run_id,
        }
        create_feedback_scores(
            langsmith_run_id=langsmith_run_id,
            scores={
                "context_precision": context_precision_score,
                "context_recall": context_recall_score,
                "faithfulness": faithfulness_score,
                "response_relevance": answer_relevancy_score,
                "completeness": answer_correctness_score,
                "hallucination_rate": (
                    round(1 - faithfulness_score, 4)
                    if faithfulness_score is not None
                    else None
                ),
            },
            metadata={
                "evaluation_run_id": run_id,
                "release_id": metadata["release_id"],
                "question_id": source_record["id"],
            },
        )
        results.append({key: normalize_value(value) for key, value in result.items()})

    return results


def main() -> None:
    run_id = build_run_id()
    records = load_answer_records()
    dataset = build_ragas_dataset(records)

    try:
        ragas_result = evaluate(
            dataset,
            metrics=build_ragas_metrics(),
            llm=build_evaluator_llm(),
            embeddings=build_evaluator_embeddings(),
            run_config=RunConfig(timeout=300, max_retries=1, max_wait=10, max_workers=1),
            batch_size=1,
            raise_exceptions=True,
        )
    except Exception as error:
        output_paths = write_error_report(run_id=run_id, error=error)
        print(f"RAGAS evaluation stopped: {type(error).__name__}: {error}")
        print(f"Saved run error to {output_paths['error_json']}")
        print(f"Saved scorecard JSON to {output_paths['scorecard_json']}")
        print(f"Saved scorecard Markdown to {output_paths['scorecard_md']}")
        raise SystemExit(1) from error

    results = add_metadata_and_gate_statuses(
        ragas_df=ragas_result.to_pandas(),
        records=records,
        run_id=run_id,
    )
    report = build_report(run_id=run_id, results=results)
    output_paths = write_report(report, output_dir=RAGA_EVAL_DIR / run_id)
    create_or_update_dataset(
        questions=report["question_results"],
        run_id=run_id,
        scorecard_path=output_paths["scorecard_json"],
    )
    create_evaluation_summary_run(
        report=report,
        result_path=output_paths["result_json"],
        scorecard_path=output_paths["scorecard_json"],
    )
    flush_langsmith()

    print(pd.DataFrame(results))
    print(f"Saved RAGAS evaluation report to {output_paths['result_json']}")
    print(f"Saved RAGAS evaluation report to {output_paths['result_yaml']}")
    print(f"Saved scorecard JSON to {output_paths['scorecard_json']}")
    print(f"Saved scorecard Markdown to {output_paths['scorecard_md']}")


if __name__ == "__main__":
    main()
