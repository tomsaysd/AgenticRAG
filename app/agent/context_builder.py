from app.retrieval.vector_store import VectorStore


def build_retrieval_context(question: str, k: int = 5):
    """Retrieve relevant Agentic RAG research chunks from ChromaDB."""
    store = VectorStore()
    results = store.search(question, k=k)

    retrieved = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        retrieved.append({
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page"),
            "url": metadata.get("url"),
            "title": metadata.get("title"),
            "content": document,
            "distance": round(float(distance), 4) if distance is not None else None,
        })

    return {"question": question, "retrieved_knowledge": retrieved}
