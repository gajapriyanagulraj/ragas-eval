from time import perf_counter

from src.llm import GroqLLM
from src.retriever import HandbookRetriever


def format_context(chunks: list[dict[str, object]]) -> str:
    return "\n\n".join(
        f"Source: {chunk['metadata'].get('source', 'unknown')}\n{chunk['content']}"
        for chunk in chunks
    )


class HandbookRAG:
    def __init__(
        self,
        retriever: HandbookRetriever | None = None,
        llm: GroqLLM | None = None,
    ):
        self.retriever = retriever or HandbookRetriever()
        self.llm = llm or GroqLLM()

    def ask(self, question: str) -> dict[str, object]:
        retrieval_start = perf_counter()
        chunks = self.retriever.retrieve(question)
        retrieval_time_ms = (perf_counter() - retrieval_start) * 1000

        context = format_context(chunks)

        generation_start = perf_counter()
        generation = self.llm.generate(question=question, context=context)
        generation_time_ms = (perf_counter() - generation_start) * 1000

        sources = sorted(
            {
                str(chunk["metadata"].get("source"))
                for chunk in chunks
                if chunk.get("metadata")
            }
        )
        return {
            "question": question,
            "answer": generation["answer"],
            "sources": sources,
            "contexts": [str(chunk["content"]) for chunk in chunks],
            "retrieved_chunks": chunks,
            "retrieval_time_ms": round(retrieval_time_ms, 2),
            "generation_time_ms": round(generation_time_ms, 2),
            "total_latency_ms": round(retrieval_time_ms + generation_time_ms, 2),
            "input_tokens": generation["input_tokens"],
            "output_tokens": generation["output_tokens"],
            "total_tokens": generation["total_tokens"],
        }
