from pathlib import Path
from urllib.parse import urlparse

from app.ingestion.loaders.documents import load_document
from app.ingestion.loaders.images import load_image
from app.ingestion.loaders.audio import load_audio
from app.ingestion.loaders.web import load_web


def ingest_source(source: str):

    # Handle web URLs
    parsed_url = urlparse(source)

    if parsed_url.scheme in ("http", "https") and parsed_url.netloc:
        return [load_web(source)]

    path = Path(source)
    extension = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    # Document formats
    if extension in {".pdf", ".docx", ".txt", ".md"}:
        return load_document(source)

    # Image formats
    elif extension in {".png", ".jpg", ".jpeg", ".webp"}:
        return [load_image(source)]

    # Audio formats
    elif extension in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}:
        return load_audio(source)

    else:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )