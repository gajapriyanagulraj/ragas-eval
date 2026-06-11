import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.config import PHOENIX_PROJECT_NAME  # noqa: E402
from src.phoenix_observability import finish_rag_trace, trace_rag_question  # noqa: E402


def main() -> None:
    with trace_rag_question(
        question_id="SMOKE",
        question="Can Phoenix receive a trace from rag-pro?",
    ) as span:
        ids = finish_rag_trace(
            span=span,
            rag_result={
                "answer": "Phoenix smoke test completed.",
                "contexts": ["Smoke test context"],
                "sources": ["phoenix_smoke_test.py"],
                "retrieval_time_ms": 1,
                "generation_time_ms": 1,
                "total_latency_ms": 2,
                "input_tokens": 5,
                "output_tokens": 5,
                "total_tokens": 10,
            },
        )

    print("Phoenix smoke test completed.")
    print(f"Project: {PHOENIX_PROJECT_NAME}")
    print(f"Trace ID: {ids['phoenix_trace_id']}")
    print("Open Phoenix at http://localhost:6006 and search for:")
    print("  employee_handbook_rag")


if __name__ == "__main__":
    main()
