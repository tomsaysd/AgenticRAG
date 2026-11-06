from pathlib import Path
from urllib.parse import urlparse

from app.retrieval.ingest import ingest_into_vector_store

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"}


def read_web_sources(path: Path):
    if not path.exists():
        return []
    urls = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parsed = urlparse(line)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            urls.append(line)
        else:
            print(f"Skipping invalid URL in {path.name}: {line}")
    return urls


def ingest_all():
    sources = []
    for folder in (DOCUMENTS_DIR, IMAGES_DIR):
        if folder.exists():
            sources.extend(sorted(
                str(path) for path in folder.iterdir()
                if path.is_file()
                and path.name.lower() != "web_sources.txt"
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ))

    sources.extend(read_web_sources(DOCUMENTS_DIR / "web_sources.txt"))
    if not sources:
        raise RuntimeError("No supported documents, images, or URLs were found.")

    successes, failures, total_chunks = [], [], 0
    for source in sources:
        try:
            count = ingest_into_vector_store(source)
            if count <= 0:
                raise RuntimeError("Source produced zero chunks.")
            successes.append(source)
            total_chunks += count
            print(f"OK: {source} ({count} chunks)")
        except Exception as exc:
            failures.append((source, str(exc)))
            print(f"FAILED: {source} — {exc}")

    print("\n--- INGESTION SUMMARY ---")
    print(f"Successful sources: {len(successes)}")
    print(f"Failed sources: {len(failures)}")
    print(f"Total chunks indexed: {total_chunks}")
    from app.retrieval.vector_store import VectorStore
    print(f"ChromaDB collection count: {VectorStore().collection.count()}")
    if failures:
        print("\nFailed sources:")
        for source, error in failures:
            print(f"- {source}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    ingest_all()
