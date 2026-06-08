from pathlib import Path
from typing import Any

import yaml

from src.config import MANIFEST_PATH


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a YAML mapping: {path}")

    validate_manifest(data)
    return data


def validate_manifest(manifest: dict[str, Any]) -> None:
    required_gates = {
        "context_relevance",
        "faithfulness",
        "response_relevance",
        "completeness",
        "hallucination_rate",
    }
    gates = manifest.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("Manifest must define a gates mapping.")

    missing = sorted(required_gates - set(gates))
    if missing:
        raise ValueError(f"Manifest gates missing required keys: {', '.join(missing)}")

    for name in sorted(required_gates):
        value = gates[name]
        if not isinstance(value, int | float):
            raise ValueError(f"Manifest gate {name} must be a number.")
        if not 0 <= float(value) <= 1:
            raise ValueError(f"Manifest gate {name} must be between 0 and 1.")


def evaluation_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "release_id": manifest["version"],
        "environment": manifest["environment"],
        "dataset": manifest["dataset"]["path"],
        "dataset_version": manifest["dataset"].get("version"),
        "embedding_model": manifest["retrieval"]["embedding_model"],
        "llm": manifest["model"]["name"],
        "chunk_size": manifest["retrieval"]["chunk_size"],
        "chunk_overlap": manifest["retrieval"]["chunk_overlap"],
        "top_k": manifest["retrieval"]["top_k"],
        "collection_name": manifest["retrieval"]["collection_name"],
        "prompt": manifest["prompt"],
    }


def quality_gates(manifest: dict[str, Any]) -> dict[str, float]:
    gates = manifest["gates"]
    return {
        "faithfulness": gates["faithfulness"],
        "answer_relevancy": gates["response_relevance"],
        "answer_correctness": gates["completeness"],
        "context_precision": gates["context_relevance"],
        "context_recall": gates["context_relevance"],
        "hallucination_rate": gates["hallucination_rate"],
    }
