import hashlib

import chromadb

from app.embeddings.embedder import Embedder


class VectorStore:
    """Persistent ChromaDB store for the Agentic RAG research corpus."""

    def __init__(self, collection_name: str = "agentic_rag_knowledge"):
        self.client = chromadb.PersistentClient(path="data/chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = Embedder()

    def add_documents(self, chunks):
        if not chunks:
            return 0

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)
        ids = []

        for index, chunk in enumerate(chunks):
            source_identity = chunk.metadata.get("file_path") or chunk.source
            page = chunk.metadata.get("page", "no_page")
            chunk_index = chunk.metadata.get("chunk_index", index)
            unique_key = f"{source_identity}|{page}|{chunk_index}"
            ids.append(hashlib.sha256(unique_key.encode("utf-8")).hexdigest())

        metadatas = []
        for chunk in chunks:
            metadata = {
                key: value
                for key, value in chunk.metadata.items()
                if value is not None and isinstance(value, (str, int, float, bool))
            }
            metadata.update({"source": chunk.source, "modality": chunk.modality})
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(self, query: str, k: int = 5):
        if self.collection.count() == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_embedding = self.embedder.embed_text(query)
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
