from pathlib import Path
from urllib.parse import urlparse

from app.ingestion.pipeline import process_source
from app.retrieval.vector_store import VectorStore


def get_document_metadata(source: str):
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return {
            "source_type": "web",
            "url": source,
            "topic": "Agentic RAG and retrieval-augmented generation",
        }

    path = Path(source)
    return {
        "source_type": "document",
        "file_type": path.suffix.lower().lstrip("."),
        "file_path": str(path.resolve()),
        "topic": "Agentic RAG and retrieval-augmented generation",
    }


def ingest_into_vector_store(source: str):
    chunks = process_source(source)
    metadata = get_document_metadata(source)
    for chunk in chunks:
        chunk.metadata.update(metadata)

    return VectorStore().add_documents(chunks)
