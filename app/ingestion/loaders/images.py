from pathlib import Path
from langchain_community.document_loaders import UnstructuredImageLoader
from app.ingestion.normalizer import IngestedDocument

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}


def load_image(file_path: str) -> IngestedDocument:
    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    loader = UnstructuredImageLoader(str(path), mode="single")
    documents = loader.load()
    if not documents or not documents[0].page_content.strip():
        raise ValueError(f"No text could be extracted from image: {path.name}")

    return IngestedDocument(
        content=documents[0].page_content.strip(),
        source=path.name,
        modality="image",
        metadata={
            "file_type": path.suffix.lower(),
            "file_path": str(path.resolve()),
            "extraction_method": "langchain_unstructured",
            "loader": "UnstructuredImageLoader",
        },
    )
