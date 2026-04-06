from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

# Path where ChromaDB will store the vector database on disk
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/processed/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_embedding_model():
    """
    Loads the sentence-transformers embedding model.
    all-MiniLM-L6-v2 is fast, free, and works great for
    semantic similarity search. Downloads on first use (~90MB).

    Returns:
        HuggingFaceEmbeddings instance
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings


def embed_and_store(chunks: list, collection_name: str = "documents") -> Chroma:
    """
    Takes text chunks, converts each one to a vector embedding,
    and stores them all in ChromaDB for later retrieval.

    Think of this like creating an index — you do it once per document
    and then search it many times.

    Args:
        chunks: List of document chunks from chunker.py
        collection_name: Name for this group of documents in ChromaDB

    Returns:
        ChromaDB vector store instance
    """
    embeddings = get_embedding_model()

    print(f"Embedding {len(chunks)} chunks and storing in ChromaDB...")
    print(f"Database path: {CHROMA_DB_PATH}")

    # Create or add to existing ChromaDB collection
    # If the collection already exists, new chunks are added to it
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DB_PATH
    )

    print(f"Successfully stored {len(chunks)} chunks in ChromaDB")
    return vectorstore


def load_vectorstore(collection_name: str = "documents") -> Chroma:
    """
    Loads an existing ChromaDB vector store from disk.
    Use this when the app starts up to load previously indexed documents.

    Args:
        collection_name: Name of the collection to load

    Returns:
        ChromaDB vector store instance ready for searching
    """
    embeddings = get_embedding_model()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    return vectorstore


def delete_vectorstore():
    """
    Deletes the entire ChromaDB database from disk.
    Use this to reset the system and start fresh.
    """
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        print(f"Deleted ChromaDB at: {CHROMA_DB_PATH}")
    else:
        print("No ChromaDB found to delete")


def get_document_count() -> int:
    """
    Returns how many chunks are currently stored in ChromaDB.
    Useful for showing the user how many documents are indexed.

    Returns:
        Number of chunks in the vector store
    """
    try:
        vectorstore = load_vectorstore()
        count = vectorstore._collection.count()
        return count
    except Exception:
        return 0
