from app.retrieval.vector_store import VectorStore


def main():
    store = VectorStore()
    if store.collection.count() == 0:
        raise SystemExit("Knowledge base is empty. Run rebuild_knowledge_base.py first.")
    questions = [
        "What is Agentic RAG?",
        "How do planning and reflection work in Agentic RAG?",
        "What are the limitations of Agentic RAG?",
    ]
    for question in questions:
        results = store.search(question, k=5)
        docs = results.get("documents", [[]])[0]
        print(f"\nQUESTION: {question}\nRetrieved: {len(docs)}")
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
            print(f"- {metadata.get('source')} | page={metadata.get('page')} | distance={distance:.4f}")


if __name__ == "__main__":
    main()
