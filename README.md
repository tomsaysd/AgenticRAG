# Agentic RAG Research Assistant (Traditional RAG)

This project is a **traditional, single-pass Retrieval-Augmented Generation (RAG) system** whose knowledge base is about **Agentic RAG**. The application does not implement an autonomous agent loop. It embeds the user's question, retrieves relevant chunks from ChromaDB, and sends those chunks to an LLM to generate a cited answer.

## Preserved technology stack

- Streamlit interface
- LangChain Community document loaders and `RecursiveCharacterTextSplitter`
- Sentence Transformers `all-MiniLM-L6-v2` embeddings (384-dimensional vectors)
- ChromaDB persistent vector store with cosine similarity
- OpenAI Responses API; model defaults to `gpt-6-luna`
- Existing PDF, DOCX, TXT/Markdown, image and web URL ingestion paths

Chunking defaults are set to 900 characters with 150 characters overlap for the research corpus. The embedding model and underlying vector-store technology are unchanged.

## Project layout

- `data/documents/`: supplied research PDFs and `web_sources.txt`
- `data/images/`: supplied Agentic RAG diagrams
- `app/ingestion/`: loaders, normalization, routing and chunking
- `app/embeddings/`: Sentence Transformers embedder
- `app/retrieval/`: ChromaDB and ingestion orchestration
- `app/agent/`: RAG context assembly and answer generation (the historical folder name is retained)
- `rebuild_knowledge_base.py`: removes the known old collections and rebuilds the new collection
- `verify_knowledge_base.py`: checks collection names and indexed chunks

## Setup on Windows

Run commands from the project root (`AgenticRAG`).

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` in the project root using `.env.example`, then add your own API key:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-6-luna
```

Never upload or commit `.env`.

## Rebuild the knowledge base

```cmd
python rebuild_knowledge_base.py
```

This deletes only these known legacy/target collections from `data/chroma_db`:

- `financial_knowledge`
- `research_documents`
- `agentic_rag_knowledge` (to make rebuilds repeatable)

It leaves any other collections untouched. It then creates `agentic_rag_knowledge` and ingests supported files in `data/documents/`, supported images in `data/images/`, and valid URLs listed one per line in `data/documents/web_sources.txt`.

The script exits with a non-zero status if any source fails. Review the failure list before treating the rebuild as complete.

## OCR for scanned PDFs

The PDF loader extracts text with PyPDFLoader and attempts OCR for pages without extractable text if PyMuPDF, Pillow, pytesseract and the Tesseract executable are installed. On Windows, install Tesseract OCR separately and ensure `tesseract.exe` is on PATH. OCR output must be checked; it can contain recognition errors.

## Run the application

```cmd
python -m streamlit run app/ui.py
```

Ask questions about Agentic RAG concepts, architectures, retrieval strategies, planning, reflection, evaluation, applications and limitations.

## Verify retrieval

```cmd
python verify_knowledge_base.py
python -m tests.test_retrieval
```

The app prints retrieved passages in the terminal for debugging. The answer cites the original filename and PDF page when available, or the page title and URL for web sources.

## Notes and limitations

- WebBaseLoader extracts page text; JavaScript-heavy, blocked or dynamically rendered pages may fail or yield incomplete content.
- Images are ingested using the existing Unstructured image loader. OCR quality depends on the image and installed system dependencies.
- The supplied `survey_on_RAG.pdf` was a duplicate of `survey_on_Agentic_Rag.pdf` based on extracted text, so the duplicate copy was removed from this project package.
- The supplied `ARAG_usecase.pdf` and `how_rag_works.pdf` had no extractable text in the initial PDF text check. They are retained; the OCR fallback may extract their content when rebuilt on a machine with Tesseract installed.
- Retrieval quality must be evaluated with actual questions. A successful ingestion does not itself prove that every retrieved chunk is relevant.
