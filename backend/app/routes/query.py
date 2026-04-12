from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.retriever import retrieve_relevant_chunks, format_context_for_prompt
from app.services.llm import generate_answer_hf_api, generate_answer_endpoint, generate_answer_local_ragnarok, MODEL_ID
from app.services import state

router = APIRouter()


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

    # Step 4: Generate answer — route to correct model based on toggle
    try:
        if state.model_mode == "local_ragnarok":
            # Local pipeline handles retrieval + generation — return its full response
            local_result = generate_answer_local_ragnarok(request.question)
            return QueryResponse(
                question=request.question,
                answer=local_result.get("answer", "No answer returned."),
                sources=local_result.get("sources", sources),
                chunks_searched=local_result.get("chunks_searched", len(results))
            )
        elif state.model_mode == "ragnarok_tuned":
            answer = generate_answer_endpoint(request.question, context)
        else:
            answer = generate_answer_hf_api(request.question, context, model_id=MODEL_ID)
    except Exception as e:
        import traceback
        traceback.print_exc()
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
