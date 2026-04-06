from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.retriever import retrieve_relevant_chunks, format_context_for_prompt
from app.services.llm import generate_answer_hf_api
from dotenv import load_dotenv
import os

load_dotenv()

router = APIRouter()

# Set to True to use local LLaMA model
# Set to False to use HuggingFace Inference API (better for deployment)
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"


class QueryRequest(BaseModel):
    """Request body for the /query endpoint"""
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response body from the /query endpoint"""
    question: str
    answer: str
    sources: list
    chunks_searched: int


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    API endpoint: POST /query

    This is the main RAG endpoint. Takes a user question,
    finds the most relevant document chunks, and generates
    an answer using LLaMA 3.

    Full RAG flow:
    1. Embed the user question
    2. Search ChromaDB for most similar chunks
    3. Format chunks into context
    4. Send context + question to LLaMA 3
    5. Return answer + source citations

    Args:
        request: JSON body with "question" field

    Returns:
        JSON with answer, sources, and metadata
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Step 1 & 2: Retrieve relevant chunks from ChromaDB
    try:
        results = retrieve_relevant_chunks(
            query=request.question,
            k=request.top_k
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}. Make sure you have uploaded documents first."
        )

    if not results:
        return QueryResponse(
            question=request.question,
            answer="No relevant documents found. Please upload some PDF documents first using the /upload endpoint.",
            sources=[],
            chunks_searched=0
        )

    # Step 3: Format context for the prompt
    context, sources = format_context_for_prompt(results)

    # Step 4: Generate answer with LLaMA 3
    try:
        if USE_LOCAL_MODEL:
            from app.services.llm import generate_answer
            answer = generate_answer(request.question, context)
        else:
            answer = generate_answer_hf_api(request.question, context)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {str(e)}"
        )

    # Step 5: Return the answer with sources
    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        chunks_searched=len(results)
    )
