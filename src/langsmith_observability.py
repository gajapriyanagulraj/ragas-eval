from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config import (
    LANGCHAIN_API_KEY,
    LANGCHAIN_PROJECT,
    LANGCHAIN_TRACING_V2,
    LANGSMITH_DATASET_NAME,
    MANIFEST_PATH,
)
from src.manifest import evaluation_metadata, load_manifest

try:
    from langsmith import Client
except ImportError:  # pragma: no cover - optional dependency guard
    Client = None  # type: ignore[assignment]

_LANGSMITH_CLIENT: Any | None = None


def is_langsmith_enabled() -> bool:
    return bool(
        Client
        and LANGCHAIN_API_KEY
        and LANGCHAIN_TRACING_V2 in {"true", "1", "yes"}
    )


def _client() -> Any | None:
    global _LANGSMITH_CLIENT
    if not is_langsmith_enabled():
        return None
    if _LANGSMITH_CLIENT is None:
        _LANGSMITH_CLIENT = Client()
    return _LANGSMITH_CLIENT


def flush_langsmith() -> None:
    client = _client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as error:  # pragma: no cover - network/provider dependent
        print(f"LangSmith flush skipped: {error}", flush=True)


def _manifest_metadata() -> dict[str, Any]:
    manifest = load_manifest()
    metadata = evaluation_metadata(manifest)
    return {
        "release_id": metadata["release_id"],
        "manifest": str(MANIFEST_PATH.relative_to(MANIFEST_PATH.parents[1])),
        "llm": metadata["llm"],
        "embedding_model": metadata["embedding_model"],
        "chunk_size": metadata["chunk_size"],
        "chunk_overlap": metadata["chunk_overlap"],
        "top_k": metadata["top_k"],
    }


def create_rag_trace(
    *,
    question_id: str,
    question: str,
    rag_result: dict[str, Any],
) -> str | None:
    client = _client()
    if client is None:
        return None

    run_id = uuid4()
    metadata = {
        **_manifest_metadata(),
        "question_id": question_id,
        "contexts_retrieved": len(rag_result.get("contexts", [])),
        "retrieval_latency_ms": rag_result.get("retrieval_time_ms"),
        "generation_latency_ms": rag_result.get("generation_time_ms"),
        "total_latency_ms": rag_result.get("total_latency_ms"),
        "prompt_tokens": rag_result.get("input_tokens"),
        "completion_tokens": rag_result.get("output_tokens"),
        "total_tokens": rag_result.get("total_tokens"),
    }

    try:
        client.create_run(
            id=run_id,
            project_name=LANGCHAIN_PROJECT,
            name="employee_handbook_rag",
            run_type="chain",
            inputs={"question": question},
            outputs={
                "answer": rag_result.get("answer"),
                "contexts": rag_result.get("contexts", []),
                "sources": rag_result.get("sources", []),
            },
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            extra={"metadata": metadata},
            tags=["rag", "employee-handbook"],
        )
    except Exception as error:  # pragma: no cover - network/provider dependent
        print(f"LangSmith trace skipped for {question_id}: {error}", flush=True)
        return None

    return str(run_id)


def ensure_rag_trace(
    *,
    question_record: dict[str, Any],
    evaluation_run_id: str,
) -> str | None:
    existing_run_id = question_record.get("langsmith_run_id")
    if existing_run_id:
        return str(existing_run_id)

    client = _client()
    if client is None:
        return None

    run_id = uuid4()
    metadata = {
        **_manifest_metadata(),
        "evaluation_run_id": evaluation_run_id,
        "question_id": question_record.get("id"),
        "contexts_retrieved": len(question_record.get("contexts", [])),
        "retrieval_latency_ms": question_record.get("retrieval_time_ms"),
        "generation_latency_ms": question_record.get("generation_time_ms"),
        "total_latency_ms": question_record.get("total_latency_ms"),
        "prompt_tokens": question_record.get("input_tokens"),
        "completion_tokens": question_record.get("output_tokens"),
        "total_tokens": question_record.get("total_tokens"),
    }

    try:
        client.create_run(
            id=run_id,
            project_name=LANGCHAIN_PROJECT,
            name="employee_handbook_rag",
            run_type="chain",
            inputs={"question": question_record.get("question")},
            outputs={
                "answer": question_record.get("answer"),
                "contexts": question_record.get("contexts", []),
                "sources": question_record.get("sources", []),
            },
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            extra={"metadata": metadata},
            tags=["rag", "employee-handbook", "ragas-evaluation"],
        )
    except Exception as error:  # pragma: no cover - network/provider dependent
        print(
            f"LangSmith trace skipped for {question_record.get('id')}: {error}",
            flush=True,
        )
        return None

    return str(run_id)


def create_feedback_scores(
    *,
    langsmith_run_id: str | None,
    scores: dict[str, float | None],
    metadata: dict[str, Any],
) -> None:
    if not langsmith_run_id:
        return

    client = _client()
    if client is None:
        return

    for key, score in scores.items():
        if score is None:
            continue
        try:
            client.create_feedback(
                run_id=langsmith_run_id,
                key=key,
                score=score,
                source_info={"source": "ragas"},
                comment=f"{key} from RAGAS evaluation",
                extra={"metadata": metadata},
            )
        except Exception as error:  # pragma: no cover - network/provider dependent
            print(f"LangSmith feedback skipped for {key}: {error}", flush=True)


def create_or_update_dataset(
    *,
    questions: list[dict[str, Any]],
    run_id: str,
    scorecard_path: Path,
) -> None:
    client = _client()
    if client is None:
        return

    metadata = {
        **_manifest_metadata(),
        "evaluation_run_id": run_id,
        "scorecard": str(scorecard_path),
    }

    try:
        if not client.has_dataset(dataset_name=LANGSMITH_DATASET_NAME):
            client.create_dataset(
                dataset_name=LANGSMITH_DATASET_NAME,
                description="Employee handbook RAG evaluation questions and answers.",
                metadata=metadata,
            )

        existing_examples = {
            example.inputs.get("id")
            for example in client.list_examples(dataset_name=LANGSMITH_DATASET_NAME)
        }
        for question in questions:
            question_id = question.get("id")
            if question_id in existing_examples:
                continue
            client.create_example(
                dataset_name=LANGSMITH_DATASET_NAME,
                inputs={
                    "id": question_id,
                    "question": question.get("question"),
                },
                outputs={
                    "expected_answer": question.get("expected_answer"),
                    "generated_answer": question.get("generated_answer"),
                },
                metadata={
                    **metadata,
                    "question_id": question_id,
                    "status": question.get("status"),
                },
            )
    except Exception as error:  # pragma: no cover - network/provider dependent
        print(f"LangSmith dataset sync skipped: {error}", flush=True)


def create_evaluation_summary_run(
    *,
    report: dict[str, Any],
    result_path: Path,
    scorecard_path: Path,
) -> None:
    client = _client()
    if client is None:
        return

    metadata = {
        **_manifest_metadata(),
        "evaluation_run_id": report["run_id"],
        "result": str(result_path),
        "scorecard": str(scorecard_path),
        "status": report["summary"]["overall_status"],
    }

    run_id = uuid4()
    try:
        client.create_run(
            id=run_id,
            project_name=LANGCHAIN_PROJECT,
            name="employee_rag_ragas_evaluation",
            run_type="chain",
            inputs={
                "release_id": report["release_id"],
                "questions_evaluated": report["summary"]["questions_evaluated"],
            },
            outputs={
                "status": report["summary"]["overall_status"],
                "retrieval": report["retrieval"],
                "generation": report["generation"],
                "rag_system": report["rag_system"],
                "performance": {
                    **report["performance"],
                    **report["tokens"],
                },
                "gates": report["gates"],
            },
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            extra={"metadata": metadata},
            tags=["ragas", "scorecard", "employee-handbook"],
        )
    except Exception as error:  # pragma: no cover - network/provider dependent
        print(f"LangSmith evaluation summary skipped: {error}", flush=True)
        return

    summary_scores = {
        "context_precision": report["retrieval"].get("context_precision"),
        "context_recall": report["retrieval"].get("context_recall"),
        "faithfulness": report["generation"].get("faithfulness"),
        "response_relevance": report["generation"].get("response_relevance"),
        "completeness": report["rag_system"].get("completeness"),
        "hallucination_rate": report["rag_system"].get("hallucination_rate"),
    }
    create_feedback_scores(
        langsmith_run_id=str(run_id),
        scores=summary_scores,
        metadata=metadata,
    )
