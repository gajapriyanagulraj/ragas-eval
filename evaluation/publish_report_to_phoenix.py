import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.config import PHOENIX_PROJECT_NAME, RAGA_EVAL_DIR  # noqa: E402
from src.phoenix_observability import (  # noqa: E402
    record_saved_question_report,
    record_saved_scorecard,
)


def resolve_run_dir(run_id: str) -> Path:
    run_dir = RAGA_EVAL_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run report folder not found: {run_dir}")
    return run_dir


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required report file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def publish_report(run_id: str) -> None:
    run_dir = resolve_run_dir(run_id)
    result_path = run_dir / "result.json"
    scorecard_path = run_dir / "scorecard.json"

    result = load_json(result_path)
    scorecard = load_json(scorecard_path)

    for question in result.get("questions", []):
        record_saved_question_report(run_id=run_id, question=question)

    record_saved_scorecard(
        scorecard=scorecard,
        result_path=result_path,
        scorecard_path=scorecard_path,
    )

    print("Published report to Phoenix.")
    print(f"Project: {PHOENIX_PROJECT_NAME}")
    print(f"Run ID: {run_id}")
    print("Open Phoenix and filter by:")
    print(f"  run_id = {run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish an existing RAGAS report run to Arize Phoenix."
    )
    parser.add_argument(
        "run_id",
        help="Report run id, for example run-20260609T170918Z.",
    )
    args = parser.parse_args()
    publish_report(run_id=args.run_id)


if __name__ == "__main__":
    main()
