
from langchain_community.document_loaders import WebBaseLoader

from app.ingestion.normalizer import IngestedDocument


def load_web(url: str) -> IngestedDocument:

    loader = WebBaseLoader(
        web_paths=(url,)
    )

    documents = loader.load()

    if not documents:
        raise ValueError("No content could be extracted from URL.")

    document = documents[0]

    content = document.page_content.strip()

    if not content:
        raise ValueError("No text could be extracted from URL.")

    return IngestedDocument(
        content=content,
        source=url,
        modality="web",
        metadata={
            "url": url,
            "title": document.metadata.get("title", url),
            "extraction_method": "langchain_webbaseloader"
        }
    )
