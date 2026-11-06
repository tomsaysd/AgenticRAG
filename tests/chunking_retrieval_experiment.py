"""Compare the project's research-oriented chunking settings on a chosen PDF."""
import sys
from pathlib import Path
import numpy as np

from app.ingestion.loaders.documents import load_document
from app.ingestion.chunker import chunk_document
from app.embeddings.embedder import Embedder

PDF_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/documents/survey_on_Agentic_Rag.pdf")
QUESTIONS = [
    "What is Agentic RAG and how does it differ from traditional RAG?",
    "What planning and reasoning capabilities are used in Agentic RAG?",
    "How does iterative retrieval improve answer quality?",
    "What are the limitations and evaluation challenges of Agentic RAG?",
]
CONFIGURATIONS = [(500, 75), (900, 150), (1200, 200)]


def main():
    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found: {PDF_PATH}")
    documents = load_document(str(PDF_PATH))
    embedder = Embedder()
    for size, overlap in CONFIGURATIONS:
        chunks = [chunk for doc in documents for chunk in chunk_document(doc, size, overlap)]
        vectors = np.asarray(embedder.embed_documents([chunk.content for chunk in chunks]))
        print(f"\nChunk size={size}, overlap={overlap}, chunks={len(chunks)}")
        for question in QUESTIONS:
            q = np.asarray(embedder.embed_text(question))
            scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q) + 1e-12)
            top = np.argsort(scores)[-3:][::-1]
            print(f"\nQ: {question}")
            for idx in top:
                print(f"similarity={scores[idx]:.4f} | {chunks[idx].source} | page={chunks[idx].metadata.get('page')}")
                print(chunks[idx].content[:300].replace("\n", " "))


if __name__ == "__main__":
    main()
