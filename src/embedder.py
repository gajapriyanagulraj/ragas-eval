import time

import voyageai

from src.config import EMBED_BATCH_DELAY_SECONDS, EMBED_BATCH_SIZE, VOYAGE_API_KEY, VOYAGE_EMBED_MODEL


class VoyageEmbedder:
    def __init__(self, api_key: str | None = VOYAGE_API_KEY, model: str = VOYAGE_EMBED_MODEL):
        if not api_key or api_key == "your_voyage_api_key":
            raise RuntimeError("Set VOYAGE_API_KEY in .env before generating embeddings.")
        self.client = voyageai.Client(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            embeddings.extend(self._embed(batch, input_type="document"))
            if start + EMBED_BATCH_SIZE < len(texts):
                time.sleep(EMBED_BATCH_DELAY_SECONDS)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type="query")[0]

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type,
        )
        return response.embeddings
