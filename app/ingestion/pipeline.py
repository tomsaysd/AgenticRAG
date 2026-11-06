from app.ingestion.router import ingest_source
from app.ingestion.chunker import chunk_document


def process_source(source: str):
    # Load the source
    documents = ingest_source(source)

    # Split every loaded document into chunks
    all_chunks = []

    for document in documents:
        chunks = chunk_document(document)
        all_chunks.extend(chunks)

    return all_chunks