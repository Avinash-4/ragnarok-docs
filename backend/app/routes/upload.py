from fastapi import APIRouter, UploadFile, File, HTTPException
from app.utils.chunker import load_and_chunk_pdf
from app.services.embedder import embed_and_store, get_document_count
import os
import shutil

router = APIRouter()

# Folder where uploaded PDFs are saved temporarily
UPLOAD_DIR = "./data/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    API endpoint: POST /upload

    Accepts a PDF file upload, saves it to disk, chunks it,
    embeds all chunks, and stores them in ChromaDB.

    This is the INDEXING step — you call this once per document.
    After this, the document is searchable via the /query endpoint.

    Args:
        file: The uploaded PDF file

    Returns:
        JSON with upload status, chunk count, and document count
    """
    # Only accept PDF files
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file."
        )

    # Save the uploaded file to disk
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Saved uploaded file: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Chunk the PDF
    try:
        chunks = load_and_chunk_pdf(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    # Embed and store in ChromaDB
    try:
        embed_and_store(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed document: {str(e)}")

    # Get total document count after indexing
    total_chunks = get_document_count()

    return {
        "status": "success",
        "message": f"Document '{file.filename}' indexed successfully",
        "chunks_created": len(chunks),
        "total_chunks_in_db": total_chunks,
        "filename": file.filename
    }


@router.get("/documents")
async def list_documents():
    """
    API endpoint: GET /documents

    Returns a list of all PDF files that have been uploaded
    and how many total chunks are in the vector database.

    Returns:
        JSON with list of uploaded files and total chunk count
    """
    # List all PDFs in the upload directory
    if not os.path.exists(UPLOAD_DIR):
        return {"documents": [], "total_chunks": 0}

    pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
    total_chunks = get_document_count()

    return {
        "documents": pdf_files,
        "document_count": len(pdf_files),
        "total_chunks_in_db": total_chunks
    }
