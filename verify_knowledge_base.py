"""Check collection contents and ensure legacy financial collections are absent."""
import chromadb

from app.retrieval.vector_store import VectorStore

client = chromadb.PersistentClient(path="data/chroma_db")
collection_names = {collection.name for collection in client.list_collections()}
print("Collections:", sorted(collection_names))

for legacy in ("financial_knowledge", "research_documents"):
    if legacy in collection_names:
        raise SystemExit(f"ERROR: legacy collection still exists: {legacy}")

collection = VectorStore().collection
print("Agentic RAG chunks:", collection.count())
if collection.count() == 0:
    raise SystemExit("ERROR: Agentic RAG collection is empty.")

sample = collection.get(limit=min(10, collection.count()), include=["metadatas"])
print("Sample sources:")
for metadata in sample.get("metadatas", []):
    print("-", metadata.get("source"), "page=", metadata.get("page"), "url=", metadata.get("url"))
print("Verification passed.")
