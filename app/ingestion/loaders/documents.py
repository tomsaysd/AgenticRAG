from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from app.ingestion.normalizer import IngestedDocument


def _load_pdf_with_ocr_fallback(path: Path):
    """Extract PDF text page-by-page; OCR image-only pages when dependencies exist."""
    pages = PyPDFLoader(str(path), mode="page").load()
    try:
        import fitz
        import pytesseract
        from PIL import Image
        from io import BytesIO
    except ImportError:
        return pages

    for page_index, document in enumerate(pages):
        if document.page_content.strip():
            continue
        pdf = fitz.open(str(path))
        page = pdf.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.open(BytesIO(pixmap.tobytes("png")))
        extracted = pytesseract.image_to_string(image).strip()
        if extracted:
            document.page_content = extracted
            document.metadata["extraction_method"] = "pypdf_plus_tesseract_ocr"
        pdf.close()
    return pages


def load_document(file_path: str) -> list[IngestedDocument]:
    path = Path(file_path)
    extension = path.suffix.lower()
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if extension == ".pdf":
        documents = _load_pdf_with_ocr_fallback(path)
        loader_name = "PyPDFLoader (with optional Tesseract OCR fallback)"
    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))
        documents = loader.load()
        loader_name = type(loader).__name__
    elif extension in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
        documents = loader.load()
        loader_name = type(loader).__name__
    else:
        raise ValueError(f"Unsupported document format: {extension}")

    loaded = []
    total_pages = len(documents)
    for index, doc in enumerate(documents):
        content = doc.page_content.strip()
        if not content:
            continue
        metadata = {
            "file_type": extension,
            "file_path": str(path.resolve()),
            "loader": loader_name,
        }
        if extension == ".pdf":
            metadata["page"] = int(doc.metadata.get("page", index)) + 1
            metadata["page_count"] = total_pages
            if doc.metadata.get("extraction_method"):
                metadata["extraction_method"] = doc.metadata["extraction_method"]
        loaded.append(IngestedDocument(
            content=content,
            source=path.name,
            modality="document",
            metadata=metadata,
        ))

    if not loaded:
        raise ValueError(
            f"No extractable text found in {path.name}. For scanned PDFs, install "
            "PyMuPDF, pytesseract and Tesseract OCR, then retry."
        )
    return loaded
