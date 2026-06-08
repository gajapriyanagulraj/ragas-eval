from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_document(document: dict[str, str]) -> list[dict[str, object]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(document["content"])

    return [
        {
            "id": f"{document['source']}:chunk-{index + 1:04d}",
            "content": chunk,
            "metadata": {
                "source": document["source"],
                "chunk_index": index,
            },
        }
        for index, chunk in enumerate(chunks)
    ]
