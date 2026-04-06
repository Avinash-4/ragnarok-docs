# 🗡️ Ragnarok-Docs

> Enterprise Document Intelligence System powered by LLaMA 3.1, RAG, and ChromaDB

Ask questions about any PDF document and get accurate answers with exact page citations — powered by Meta's open source LLaMA 3.1 8B model and a full Retrieval Augmented Generation pipeline.

---

## 🚀 Live Demo

> Upload a PDF → Ask a question → Get a cited answer from LLaMA 3.1

**API:** `https://your-railway-url.railway.app`  
**Docs:** `https://your-railway-url.railway.app/docs`

---

## 🧠 What This Project Demonstrates

This project was built as a portfolio piece to demonstrate real-world AI engineering skills:

- **Open Source LLMs** — Running Meta's LLaMA 3.1 8B Instruct via HuggingFace
- **RAG Pipeline** — Full Retrieval Augmented Generation from scratch
- **Vector Database** — ChromaDB for semantic search over documents
- **Fine-Tuning** — QLoRA fine-tuning of LLaMA 3 (Phase 3)
- **Production API** — FastAPI backend with streaming responses
- **Evaluation** — RAGAS metrics for measuring RAG quality

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│                     User Question                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend (:8000)                     │
│                                                         │
│   POST /api/upload          POST /api/query             │
│        │                         │                      │
│        ▼                         ▼                      │
│   ┌─────────┐            ┌──────────────┐               │
│   │ Chunker │            │  Retriever   │               │
│   │ PyPDF   │            │  ChromaDB    │               │
│   │ 512 tok │            │  Top-5 chunks│               │
│   └────┬────┘            └──────┬───────┘               │
│        │                        │                       │
│        ▼                        ▼                       │
│   ┌─────────┐            ┌──────────────┐               │
│   │Embedder │            │  LLaMA 3.1   │               │
│   │MiniLM   │            │  8B Instruct │               │
│   │ vectors │            │  HF API      │               │
│   └────┬────┘            └──────┬───────┘               │
│        │                        │                       │
│        ▼                        ▼                       │
│   ┌─────────┐            ┌──────────────┐               │
│   │ChromaDB │            │Answer + Cited│               │
│   │ on disk │            │   Sources    │               │
│   └─────────┘            └──────────────┘               │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Meta LLaMA 3.1 8B Instruct |
| Fine-tuning | QLoRA via PEFT + bitsandbytes |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Backend API | FastAPI + Uvicorn |
| PDF Processing | PyPDF |
| Evaluation | RAGAS |
| Frontend | React + Tailwind (Phase 3) |

---

## 📁 Project Structure
```
ragnarok-docs/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── routes/
│   │   │   ├── upload.py        # POST /api/upload
│   │   │   └── query.py         # POST /api/query
│   │   ├── services/
│   │   │   ├── embedder.py      # ChromaDB vector storage
│   │   │   ├── retriever.py     # Semantic search
│   │   │   └── llm.py           # LLaMA 3.1 inference
│   │   └── utils/
│   │       └── chunker.py       # PDF loading and chunking
│   ├── Dockerfile
│   └── requirements.txt
├── notebooks/
│   ├── 01_rag_pipeline_test.ipynb    # End-to-end RAG test
│   └── 02_finetuning_qlora.ipynb     # QLoRA fine-tuning
├── data/
│   ├── raw/                     # Uploaded PDFs
│   └── processed/               # ChromaDB vector store
├── models/                      # Fine-tuned LoRA adapters
├── evaluation/
│   ├── test_questions.json      # RAGAS test set
│   └── ragas_eval.py            # Evaluation script
├── .env.example                 # Environment variables template
└── README.md
```

---

## ⚙️ How to Run Locally

### Prerequisites

- Python 3.11 or higher
- Git
- A HuggingFace account with LLaMA 3.1 access

### Step 1 — Clone the repo
```bash
git clone https://github.com/yourusername/ragnarok-docs.git
cd ragnarok-docs
```

### Step 2 — Create virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r backend/requirements.txt --prefer-binary
```

> Note: If you are on Windows and get a Rust/disk space error during install,
> run: `pip install -r backend/requirements.txt --prefer-binary --cache-dir D:\pip-cache`
> replacing D: with any drive that has 5GB+ free space.

### Step 4 — Set up environment variables
```bash
cp .env.example .env
```

Open `.env` and fill in your values:
```
HUGGINGFACE_TOKEN=hf_your_token_here
MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
CHROMA_DB_PATH=./data/processed/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RESULTS=5
USE_LOCAL_MODEL=false
```

Get your HuggingFace token at: `huggingface.co/settings/tokens`

Request LLaMA 3.1 access at: `huggingface.co/meta-llama/Llama-3.1-8B-Instruct`

### Step 5 — Verify setup
```bash
python verify_llama_access.py
```

You should see:
```
ALL CHECKS PASSED. Ready to run!
```

### Step 6 — Start the server
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 7 — Test it

Open `http://localhost:8000/docs` in your browser.

You will see the Swagger UI with all endpoints. Try:

1. **POST /api/upload** — upload any PDF
2. **POST /api/query** — ask a question about it

---

## 🔌 API Reference

### Upload a document
```http
POST /api/upload
Content-Type: multipart/form-data

file: your-document.pdf
```

Response:
```json
{
  "status": "success",
  "message": "Document indexed successfully",
  "chunks_created": 92,
  "total_chunks_in_db": 92,
  "filename": "your-document.pdf"
}
```

### Ask a question
```http
POST /api/query
Content-Type: application/json

{
  "question": "What is the main contribution of this paper?",
  "top_k": 5
}
```

Response:
```json
{
  "question": "What is the main contribution of this paper?",
  "answer": "The main contribution is the Transformer architecture... (Source 1, Page 2)",
  "sources": [
    {
      "index": 1,
      "file": "paper.pdf",
      "page": 2,
      "relevance_score": 0.88,
      "preview": "The Transformer follows this overall architecture..."
    }
  ],
  "chunks_searched": 5
}
```

### List uploaded documents
```http
GET /api/documents
```

### Health check
```http
GET /health
```

---

## 📊 RAGAS Evaluation Results

Evaluated on 20 questions from the "Attention Is All You Need" paper:

| Metric | Score |
|---|---|
| Faithfulness | Coming in Phase 3 |
| Answer Relevancy | Coming in Phase 3 |
| Context Precision | Coming in Phase 3 |
| Context Recall | Coming in Phase 3 |

---

## 🧪 Fine-Tuning with QLoRA

The model was fine-tuned on a domain-specific Q&A dataset using QLoRA:

- **Base model:** meta-llama/Llama-3.1-8B-Instruct
- **Method:** QLoRA (4-bit quantization + LoRA adapters)
- **LoRA config:** r=16, alpha=32, target=q_proj,v_proj
- **Training:** Google Colab free tier T4 GPU
- **Dataset:** Coming in Phase 3

Fine-tuned adapter available at: `huggingface.co/yourusername/llama3-ragnarok-adapter`

---

## Running on a New Machine

After cloning the repo and installing dependencies,
you will need to re-upload your documents via POST /api/upload
since ChromaDB is stored locally and not tracked in Git.

Production roadmap includes migrating to Pinecone for 
persistent cloud vector storage and AWS S3 for document storage,
eliminating the need to re-upload on each new deployment.

## 🚢 Deployment

### Backend — Railway
```bash
# Connect your GitHub repo to Railway
# Railway auto-deploys on every push to main
# Set environment variables in Railway dashboard
```

### Frontend — Vercel
```bash
# Connect your GitHub repo to Vercel
# Set REACT_APP_API_URL to your Railway URL
# Vercel auto-deploys on every push
```

---

## 🗺️ Roadmap

- [x] Phase 1 — Environment setup and project scaffold
- [x] Phase 2 — RAG pipeline with LLaMA 3.1 and ChromaDB
- [ ] Phase 3 — QLoRA fine-tuning on Google Colab
- [ ] Phase 4 — React frontend with chat UI
- [ ] Phase 5 — RAGAS evaluation dashboard
- [ ] Phase 6 — Deploy to Railway and Vercel

---

## 🙋 Author

Built by Avinash chowdary Kongara as an AI engineering portfolio project.

- GitHub: github.com/Avinash-4

---

## 📄 License

MIT License — feel free to use this project as a reference for your own portfolio.
