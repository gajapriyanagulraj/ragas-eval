from src.config import TOP_K
from src.embedder import VoyageEmbedder
from src.vectordb import get_collection


class HandbookRetriever:
    def __init__(self, embedder: VoyageEmbedder | None = None, top_k: int = TOP_K):
        self.embedder = embedder or VoyageEmbedder()
        self.collection = get_collection()
        self.top_k = top_k

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict[str, object]]:
        query_embedding = self.embedder.embed_query(question)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k or self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            {
                "content": document,
                "metadata": metadata,
                "distance": distance,
            }
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]
