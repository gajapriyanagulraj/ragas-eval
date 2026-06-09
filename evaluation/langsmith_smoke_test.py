import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.config import LANGCHAIN_PROJECT  # noqa: E402
from src.langsmith_observability import flush_langsmith, is_langsmith_enabled  # noqa: E402

try:
    from langsmith import Client
except ImportError as error:  # pragma: no cover
    raise RuntimeError("Install dependencies with `pip install -r requirements.txt`.") from error


def main() -> None:
    if not is_langsmith_enabled():
        raise RuntimeError(
            "LangSmith is not enabled. Set LANGCHAIN_API_KEY, "
            "LANGCHAIN_TRACING_V2=true, and LANGCHAIN_PROJECT in .env."
        )

    client = Client()
    run_id = uuid4()
    client.create_run(
        id=run_id,
        project_name=LANGCHAIN_PROJECT,
        name="langsmith_smoke_test",
        run_type="chain",
        inputs={"message": "hello from rag-pro"},
        outputs={"status": "ok"},
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
        extra={
            "metadata": {
                "source": "evaluation/langsmith_smoke_test.py",
                "project": LANGCHAIN_PROJECT,
            }
        },
        tags=["smoke-test", "rag-pro"],
    )
    client.create_feedback(
        run_id=run_id,
        key="smoke_test_score",
        score=1.0,
        comment="LangSmith smoke test feedback.",
    )
    flush_langsmith()

    print("LangSmith smoke test sent.")
    print(f"Project: {LANGCHAIN_PROJECT}")
    print(f"Run ID: {run_id}")
    print("Open LangSmith > Tracing and look for project/run:")
    print(f"  {LANGCHAIN_PROJECT} / langsmith_smoke_test")


if __name__ == "__main__":
    main()
