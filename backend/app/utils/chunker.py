from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_and_chunk_pdf(pdf_path: str) -> list:
    """
    Loads a PDF file and splits it into smaller chunks.
    Each chunk is 512 tokens with 64 token overlap so context
    is not lost between chunks.

    Args:
        pdf_path: Full path to the PDF file

    Returns:
        List of document chunks with metadata (page number, source)
    """
    # Check the file exists before loading
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    # Load the PDF - each page becomes a document
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    # Split pages into smaller chunks for better retrieval
    # chunk_size=512 means each chunk is ~512 characters
    # chunk_overlap=64 means chunks share 64 characters to preserve context
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    # Add source filename to each chunk's metadata
    for chunk in chunks:
        chunk.metadata["source_file"] = os.path.basename(pdf_path)

    return chunks


def load_and_chunk_text(text: str, source_name: str = "manual_input") -> list:
    """
    Chunks a raw text string directly (no PDF needed).
    Useful for testing the pipeline without a PDF.

    Args:
        text: Raw text string to chunk
        source_name: Label to identify where this text came from

    Returns:
        List of document chunks
    """
    from langchain.schema import Document

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
    )

    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source_file": source_name, "page": 0}]
    )

    return chunks
