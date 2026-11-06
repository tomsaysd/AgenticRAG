
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion.normalizer import IngestedDocument


def chunk_document(
    document: IngestedDocument,
    chunk_size: int = 900,
    chunk_overlap: int = 150
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )

    chunks = splitter.create_documents(
        texts=[document.content],
        metadatas=[document.metadata]
    )

    chunked_documents = []

    for index, chunk in enumerate(chunks):
        chunked_documents.append(
            IngestedDocument(
                content=chunk.page_content,
                source=document.source,
                modality=document.modality,
                metadata={
                    **chunk.metadata,
                    "chunk_index": index,
                    "chunk_size": len(chunk.page_content)
                },
                timestamp_start=document.timestamp_start,
                timestamp_end=document.timestamp_end
            )
        )

    return chunked_documents
