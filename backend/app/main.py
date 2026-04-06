from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import upload, query

# Initialize the FastAPI app
app = FastAPI(
    title="Ragnarok-Docs API",
    description="Enterprise Document Intelligence System powered by LLaMA 3 and RAG",
    version="1.0.0"
)

# Allow the React frontend to call this API
# In production, replace * with your actual frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the route modules
app.include_router(upload.router, prefix="/api", tags=["Documents"])
app.include_router(query.router, prefix="/api", tags=["Query"])


@app.get("/")
async def root():
    """Health check endpoint — confirms the server is running"""
    return {
        "message": "Ragnarok-Docs API is running",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Detailed health check for deployment monitoring"""
    from app.services.embedder import get_document_count
    return {
        "status": "healthy",
        "indexed_chunks": get_document_count(),
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "vector_db": "ChromaDB"
    }
