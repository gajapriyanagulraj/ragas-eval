import chromadb
from chromadb.errors import NotFoundError

from src.config import CHROMA_DIR, COLLECTION_NAME


def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(name: str = COLLECTION_NAME):
    return get_chroma_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(name: str = COLLECTION_NAME):
    client = get_chroma_client()
    try:
        client.delete_collection(name=name)
    except (NotFoundError, ValueError):
        pass
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[dict[str, object]], embeddings: list[list[float]]) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding counts do not match.")

    collection = reset_collection()
    collection.add(
        ids=[str(chunk["id"]) for chunk in chunks],
        documents=[str(chunk["content"]) for chunk in chunks],
        metadatas=[dict(chunk["metadata"]) for chunk in chunks],
        embeddings=embeddings,
    )
    return len(chunks)
