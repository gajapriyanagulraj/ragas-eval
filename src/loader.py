from pathlib import Path


def load_document(path: Path) -> dict[str, str]:
    """Load a text document into the app's document format."""
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    return {
        "content": path.read_text(encoding="utf-8"),
        "source": path.name,
    }
