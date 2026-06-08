import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.chunker import chunk_document
from src.config import HANDBOOK_PATH
from src.embedder import VoyageEmbedder
from src.loader import load_document
from src.vectordb import add_chunks


def main() -> None:
    document = load_document(HANDBOOK_PATH)
    chunks = chunk_document(document)
    embeddings = VoyageEmbedder().embed_documents([str(chunk["content"]) for chunk in chunks])
    indexed_count = add_chunks(chunks, embeddings)
    print(f"All chunks indexed in ChromaDB: {indexed_count} chunks")


if __name__ == "__main__":
    main()
