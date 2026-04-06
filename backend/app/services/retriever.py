from app.services.embedder import load_vectorstore
from dotenv import load_dotenv
import os

load_dotenv()

TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))


def retrieve_relevant_chunks(query: str, k: int = None) -> list:
    """
    Takes a user question, converts it to a vector embedding,
    and finds the TOP K most similar chunks in ChromaDB.

    This is the RETRIEVAL part of RAG (Retrieval Augmented Generation).

    Args:
        query: The user's question as a string
        k: How many chunks to retrieve (defaults to TOP_K_RESULTS from .env)

    Returns:
        List of (document, similarity_score) tuples, ranked by relevance
    """
    if k is None:
        k = TOP_K_RESULTS

    print(f"Searching for: '{query}'")
    print(f"Retrieving top {k} chunks...")

    vectorstore = load_vectorstore()

    # similarity_search_with_score returns chunks AND their similarity scores
    # Score closer to 0 = more similar (it's a distance metric)
    results = vectorstore.similarity_search_with_score(query, k=k)

    print(f"Found {len(results)} relevant chunks")

    # Log which pages/sources were found
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"  Result {i+1}: {source} (page {page}) - score: {score:.4f}")

    return results


def format_context_for_prompt(results: list) -> tuple:
    """
    Takes the retrieved chunks and formats them into a clean context
    string that can be injected into the LLaMA prompt.

    Also extracts source citations so we can show the user
    exactly which documents and pages the answer came from.

    Args:
        results: List of (document, score) tuples from retrieve_relevant_chunks

    Returns:
        Tuple of (context_string, list_of_sources)
    """
    context_parts = []
    sources = []

    for i, (doc, score) in enumerate(results):
        source_file = doc.metadata.get("source_file", "unknown")
        page_num = doc.metadata.get("page", "unknown")

        # Format each chunk with its source label
        context_parts.append(
            f"[Source {i+1} - {source_file}, Page {page_num}]\n{doc.page_content}"
        )

        # Collect unique sources for citations
        source_entry = {
            "index": i + 1,
            "file": source_file,
            "page": page_num,
            "relevance_score": round(float(score), 4),
            "preview": doc.page_content[:150] + "..."
        }
        sources.append(source_entry)

    # Join all chunks into one context block
    context_string = "\n\n---\n\n".join(context_parts)

    return context_string, sources
