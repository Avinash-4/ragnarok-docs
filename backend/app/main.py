from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routes import upload, query, config
import os

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
app.include_router(config.router, prefix="/api", tags=["Config"])

# Serve the frontend static files
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/app", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


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
    from app.services import state
    return {
        "status": "healthy",
        "indexed_chunks": get_document_count(),
        "model_mode": state.model_mode,
        "vector_db": "ChromaDB"
    }
