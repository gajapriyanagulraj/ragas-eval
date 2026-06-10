from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.config import (
    MANIFEST_PATH,
    PHOENIX_COLLECTOR_ENDPOINT,
    PHOENIX_ENABLED,
    PHOENIX_PROJECT_NAME,
)
from src.manifest import evaluation_metadata, load_manifest

try:
    from opentelemetry import trace
except ImportError:  # pragma: no cover - optional dependency guard
    trace = None  # type: ignore[assignment]

try:
    from phoenix.otel import register
except ImportError:  # pragma: no cover - optional dependency guard
    register = None  # type: ignore[assignment]

_PHOENIX_READY = False
_PHOENIX_WARNED = False


def is_phoenix_enabled() -> bool:
    return PHOENIX_ENABLED in {"true", "1", "yes"}


def _warn_once(message: str) -> None:
    global _PHOENIX_WARNED
    if not _PHOENIX_WARNED:
        print(message, flush=True)
        _PHOENIX_WARNED = True


def _setup_phoenix() -> bool:
    global _PHOENIX_READY
    if _PHOENIX_READY:
        return True
    if not is_phoenix_enabled():
        return False
    if trace is None or register is None:
        _warn_once(
            "Phoenix tracing skipped: install Phoenix dependencies with "
            "`pip install -r requirements.txt`."
        )
        return False

    try:
        if PHOENIX_COLLECTOR_ENDPOINT:
            register(
                project_name=PHOENIX_PROJECT_NAME,
                endpoint=_collector_endpoint(),
            )
        else:
            register(project_name=PHOENIX_PROJECT_NAME)
    except TypeError:
        register(project_name=PHOENIX_PROJECT_NAME)
    except Exception as error:  # pragma: no cover - network/provider dependent
        _warn_once(f"Phoenix tracing skipped: {error}")
        return False

    _PHOENIX_READY = True
    return True


def _collector_endpoint() -> str:
    endpoint = str(PHOENIX_COLLECTOR_ENDPOINT).rstrip("/")
    if endpoint.endswith("/v1/traces"):
        return endpoint
    return f"{endpoint}/v1/traces"


def _tracer() -> Any | None:
    if not _setup_phoenix() or trace is None:
        return None
    return trace.get_tracer("rag-pro")


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


def _set_attributes(span: Any | None, attributes: dict[str, Any]) -> None:
    if span is None:
        return
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, str(value))


def _span_ids(span: Any | None) -> dict[str, str | None]:
    if span is None:
        return {"phoenix_trace_id": None, "phoenix_span_id": None}
    context = span.get_span_context()
    if not context or not context.is_valid:
        return {"phoenix_trace_id": None, "phoenix_span_id": None}
    return {
        "phoenix_trace_id": f"{context.trace_id:032x}",
        "phoenix_span_id": f"{context.span_id:016x}",
    }


@contextmanager
def trace_rag_question(
    *,
    question_id: str,
    question: str,
) -> Iterator[Any | None]:
    tracer = _tracer()
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span("employee_handbook_rag") as span:
        _set_attributes(
            span,
            {
                **_manifest_metadata(),
                "question_id": question_id,
                "input.question": question,
                "span.kind": "rag",
            },
        )
        yield span


def finish_rag_trace(
    *,
    span: Any | None,
    rag_result: dict[str, Any],
) -> dict[str, str | None]:
    _set_attributes(
        span,
        {
            "output.answer": rag_result.get("answer"),
            "retrieval.contexts_retrieved": len(rag_result.get("contexts", [])),
            "retrieval.latency_ms": rag_result.get("retrieval_time_ms"),
            "generation.latency_ms": rag_result.get("generation_time_ms"),
            "performance.total_latency_ms": rag_result.get("total_latency_ms"),
            "tokens.prompt": rag_result.get("input_tokens"),
            "tokens.completion": rag_result.get("output_tokens"),
            "tokens.total": rag_result.get("total_tokens"),
            "sources": rag_result.get("sources"),
        },
    )
    return _span_ids(span)


def record_question_evaluation(
    *,
    evaluation_run_id: str,
    question_record: dict[str, Any],
    scores: dict[str, float | None],
) -> None:
    tracer = _tracer()
    if tracer is None:
        return

    with tracer.start_as_current_span("ragas_question_evaluation") as span:
        _set_attributes(
            span,
            {
                **_manifest_metadata(),
                "run_id": evaluation_run_id,
                "evaluation_run_id": evaluation_run_id,
                "question_id": question_record.get("id"),
                "input.question": question_record.get("question"),
                "output.answer": question_record.get("answer"),
                "reference.answer": question_record.get("ground_truth"),
                "phoenix_trace_id": question_record.get("phoenix_trace_id"),
                "phoenix_span_id": question_record.get("phoenix_span_id"),
                **{f"ragas.{key}": value for key, value in scores.items()},
            },
        )


def record_evaluation_summary(
    *,
    report: dict[str, Any],
    result_path: Path,
    scorecard_path: Path,
) -> None:
    tracer = _tracer()
    if tracer is None:
        return

    with tracer.start_as_current_span("employee_rag_ragas_evaluation") as span:
        _set_attributes(
            span,
            {
                **_manifest_metadata(),
                "run_id": report["run_id"],
                "evaluation_run_id": report["run_id"],
                "result": str(result_path),
                "scorecard": str(scorecard_path),
                "status": report["summary"]["overall_status"],
                "questions_evaluated": report["summary"]["questions_evaluated"],
                "retrieval.context_precision": report["retrieval"].get("context_precision"),
                "retrieval.context_recall": report["retrieval"].get("context_recall"),
                "generation.faithfulness": report["generation"].get("faithfulness"),
                "generation.response_relevance": report["generation"].get("response_relevance"),
                "rag_system.completeness": report["rag_system"].get("completeness"),
                "rag_system.hallucination_rate": report["rag_system"].get("hallucination_rate"),
                "performance.avg_retrieval_latency_ms": report["performance"].get(
                    "avg_retrieval_latency_ms"
                ),
                "performance.avg_generation_latency_ms": report["performance"].get(
                    "avg_generation_latency_ms"
                ),
                "performance.avg_total_latency_ms": report["performance"].get(
                    "avg_total_latency_ms"
                ),
                "tokens.avg_prompt_tokens": report["tokens"].get("avg_prompt_tokens"),
                "tokens.avg_completion_tokens": report["tokens"].get(
                    "avg_completion_tokens"
                ),
                "tokens.avg_total_tokens": report["tokens"].get("avg_total_tokens"),
            },
        )


def record_saved_question_report(
    *,
    run_id: str,
    question: dict[str, Any],
) -> None:
    tracer = _tracer()
    if tracer is None:
        return

    retrieval = question.get("retrieval", {})
    generation = question.get("generation", {})
    rag_system = question.get("rag_system", {})
    performance = question.get("performance", {})

    with tracer.start_as_current_span("ragas_question_report") as span:
        _set_attributes(
            span,
            {
                **_manifest_metadata(),
                "run_id": run_id,
                "evaluation_run_id": run_id,
                "question_id": question.get("id"),
                "status": question.get("status"),
                "input.question": question.get("question"),
                "reference.answer": question.get("expected_answer"),
                "output.answer": question.get("generated_answer"),
                "retrieval.context_precision": retrieval.get("context_precision"),
                "retrieval.context_recall": retrieval.get("context_recall"),
                "generation.faithfulness": generation.get("faithfulness"),
                "generation.response_relevance": generation.get("response_relevance"),
                "rag_system.completeness": rag_system.get("completeness"),
                "rag_system.hallucination_rate": rag_system.get("hallucination_rate"),
                "performance.latency_ms": performance.get("latency_ms"),
                "tokens.prompt": performance.get("prompt_tokens"),
                "tokens.completion": performance.get("completion_tokens"),
                "tokens.total": performance.get("total_tokens"),
                "failure_reasons": question.get("failure_reasons", []),
            },
        )


def record_saved_scorecard(
    *,
    scorecard: dict[str, Any],
    result_path: Path,
    scorecard_path: Path,
) -> None:
    tracer = _tracer()
    if tracer is None:
        return

    metrics = scorecard.get("metrics", {})
    retrieval = metrics.get("retrieval", {})
    generation = metrics.get("generation", {})
    rag_system = metrics.get("rag_system", {})
    performance = scorecard.get("performance", {})
    release_decision = scorecard.get("release_decision", {})

    with tracer.start_as_current_span("employee_rag_report_scorecard") as span:
        _set_attributes(
            span,
            {
                **_manifest_metadata(),
                "run_id": scorecard.get("run_id"),
                "evaluation_run_id": scorecard.get("run_id"),
                "release_id": scorecard.get("release_id"),
                "status": scorecard.get("status"),
                "result": str(result_path),
                "scorecard": str(scorecard_path),
                "questions_evaluated": scorecard.get("summary", {}).get(
                    "questions_evaluated"
                ),
                "gates_passed": scorecard.get("summary", {}).get("gates_passed"),
                "gates_failed": scorecard.get("summary", {}).get("gates_failed"),
                "release.approved": release_decision.get("approved"),
                "release.blocking_issues": release_decision.get("blocking_issues", []),
                "retrieval.context_precision": retrieval.get("context_precision"),
                "retrieval.context_recall": retrieval.get("context_recall"),
                "retrieval.context_relevance": retrieval.get("context_relevance"),
                "generation.faithfulness": generation.get("faithfulness"),
                "generation.response_relevance": generation.get("response_relevance"),
                "rag_system.completeness": rag_system.get("completeness"),
                "rag_system.hallucination_rate": rag_system.get("hallucination_rate"),
                "performance.avg_retrieval_latency_ms": performance.get(
                    "avg_retrieval_latency_ms"
                ),
                "performance.avg_generation_latency_ms": performance.get(
                    "avg_generation_latency_ms"
                ),
                "performance.avg_total_latency_ms": performance.get(
                    "avg_total_latency_ms"
                ),
                "tokens.avg_prompt_tokens": performance.get("avg_prompt_tokens"),
                "tokens.avg_completion_tokens": performance.get("avg_completion_tokens"),
                "tokens.avg_total_tokens": performance.get("avg_total_tokens"),
            },
        )
