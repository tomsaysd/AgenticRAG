"""Safely rebuild the dedicated Agentic RAG collection from supplied sources."""
import chromadb

from app.retrieval.ingest_folder import ingest_all
from app.retrieval.vector_store import VectorStore

DB_PATH = "data/chroma_db"
NEW_COLLECTION = "agentic_rag_knowledge"
OLD_COLLECTIONS = ("financial_knowledge", "research_documents")


def main():
    client = chromadb.PersistentClient(path=DB_PATH)
    existing = {collection.name for collection in client.list_collections()}

    # Remove only the known legacy financial/research collections and the
    # previous target collection. Other unrelated collections are untouched.
    for name in (*OLD_COLLECTIONS, NEW_COLLECTION):
        if name in existing:
            client.delete_collection(name)
            print(f"Deleted old collection: {name}")

    VectorStore(collection_name=NEW_COLLECTION)
    print(f"Created clean collection: {NEW_COLLECTION}")

    ingest_all()
    final_count = VectorStore(collection_name=NEW_COLLECTION).collection.count()
    if final_count == 0:
        raise RuntimeError("Rebuild finished but the new collection is empty.")
    print(f"\nSUCCESS: {final_count} chunks in {NEW_COLLECTION}.")


if __name__ == "__main__":
    main()
