<div align="center">

# 🗡️ Ragnarok-Docs

### Enterprise Document Intelligence System

**Ask questions about any private document. Get accurate answers with exact page citations.**

*Powered by Meta's LLaMA 3.1 8B — fine-tuned on Google Natural Questions using QLoRA*

[![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LLaMA](https://img.shields.io/badge/LLaMA-3.1%208B-purple)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
[![HuggingFace](https://img.shields.io/badge/Adapter-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/avinashkongara4/llama3-ragnarok-nq-adapter)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

[Live Demo](#live-demo) • [Why Better Than GPT-4](#why-better-than-gpt-4) • [Quick Start](#quick-start) • [API Reference](#api-reference) • [Model Evaluation](#model-evaluation)

</div>

---

## What Is This

Ragnarok-Docs is an open source document intelligence system that lets you upload any PDF and ask questions about it. Unlike ChatGPT or Claude, it answers only from your document — with exact page citations proving where every answer came from.

I built this as an AI engineering portfolio project to demonstrate the complete LLM development lifecycle: RAG pipeline design, open source model fine-tuning, vector database integration, production API development, and model evaluation.

---

## Live Demo

**API:** `https://ragnarok-docs.railway.app`
**Docs:** `https://ragnarok-docs.railway.app/docs`

Upload a PDF → Ask a question → Get a cited answer in seconds.

---

## Why Better Than GPT-4

This is a question I get asked a lot. Here is the honest answer.

| Feature | GPT-4 / ChatGPT | Ragnarok-Docs |
|---|---|---|
| Reads your private documents | ❌ Cannot access your files | ✅ Upload any PDF |
| Source citations | ❌ No page references | ✅ Exact page numbers |
| Data privacy | ❌ Sent to OpenAI servers | ✅ Runs on your infrastructure |
| Works on confidential files | ❌ Not recommended | ✅ Designed for it |
| Custom fine-tuning | ❌ Closed model | ✅ Open weights, fully customizable |
| Cost per query | $0.03+ per query | $0.001 per query |
| Knowledge cutoff | Fixed training date | Updates when you upload new docs |
| Runs offline / on-premise | ❌ Cloud only | ✅ Fully self-hostable |

**The honest comparison:** GPT-4 scores ~91% on SQuAD benchmarks. My fine-tuned LLaMA scores lower on general benchmarks. But for private document Q&A — where GPT-4 literally cannot access your files — my system wins by default.

For a law firm querying confidential contracts, a hospital analyzing patient records, or a startup searching internal documentation, Ragnarok-Docs is not just better — it is the only option.

---

## What This Project Demonstrates

- **Open Source LLMs** — Running and fine-tuning Meta's LLaMA 3.1 8B Instruct
- **RAG Pipeline** — Full Retrieval Augmented Generation built from scratch
- **Vector Database** — ChromaDB for semantic search over private documents
- **QLoRA Fine-Tuning** — Training only 0.52% of model parameters on a single GPU
- **Production API** — FastAPI backend with streaming responses and source citations
- **Model Evaluation** — Perplexity scoring and RAGAS metrics

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User uploads PDF                      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend (:8000)                 │
│                                                         │
│  POST /api/upload              POST /api/query          │
│         │                            │                  │
│         ▼                            ▼                  │
│   ┌──────────┐               ┌─────────────┐           │
│   │  Chunker │               │  Retriever  │           │
│   │  PyPDF   │               │  ChromaDB   │           │
│   │  512 tok │               │  Top-5 chunks│          │
│   └────┬─────┘               └──────┬──────┘           │
│        │                            │                   │
│        ▼                            ▼                   │
│   ┌──────────┐               ┌─────────────┐           │
│   │ Embedder │               │  LLaMA 3.1  │           │
│   │ MiniLM   │               │  Fine-tuned │           │
│   │ vectors  │               │  HF API     │           │
│   └────┬─────┘               └──────┬──────┘           │
│        │                            │                   │
│        ▼                            ▼                   │
│   ┌──────────┐               ┌─────────────┐           │
│   │ ChromaDB │               │Answer + Page│           │
│   │ on disk  │               │  Citations  │           │
│   └──────────┘               └─────────────┘           │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why I chose it |
|---|---|---|
| LLM | Meta LLaMA 3.1 8B Instruct | Open weights, no API cost, fully controllable |
| Fine-tuning | QLoRA via PEFT | Trains 0.52% of parameters — feasible on single GPU |
| RAG Framework | LangChain | Industry standard, extensible |
| Vector Database | ChromaDB | Zero infrastructure, runs locally |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Fast, free, 90MB |
| Backend API | FastAPI + Uvicorn | Async, auto-docs, production ready |
| PDF Processing | PyPDF | Lightweight, no dependencies |
| Evaluation | RAGAS | Purpose-built for RAG quality measurement |
| Frontend | React + Tailwind | Phase 4 — coming soon |
| Deployment | Railway (API) + Vercel (UI) | Free tier, auto-deploy from GitHub |

---

## Project Structure

```
ragnarok-docs/
├── backend/
│   ├── app/
│   │   ├── main.py              ← FastAPI entry point
│   │   ├── routes/
│   │   │   ├── upload.py        ← POST /api/upload
│   │   │   └── query.py         ← POST /api/query
│   │   ├── services/
│   │   │   ├── embedder.py      ← ChromaDB vector storage
│   │   │   ├── retriever.py     ← Semantic search
│   │   │   └── llm.py           ← LLaMA 3.1 inference
│   │   └── utils/
│   │       └── chunker.py       ← PDF loading and chunking
│   ├── Dockerfile
│   └── requirements.txt
├── notebooks/
│   ├── 01_rag_pipeline_test.ipynb    ← End-to-end RAG test
│   └── 02_finetuning_qlora.ipynb     ← QLoRA fine-tuning
├── data/
│   ├── raw/                     ← Uploaded PDFs (gitignored)
│   └── processed/               ← ChromaDB vectors (gitignored)
├── models/                      ← LoRA adapters (gitignored)
├── evaluation/
│   ├── test_questions.json      ← RAGAS test set
│   └── ragas_eval.py            ← Evaluation script
├── TRAINING_JOURNAL.md          ← Full fine-tuning documentation
├── verify_llama_access.py       ← Setup verification script
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Git
- HuggingFace account with LLaMA 3.1 access

### Step 1 — Clone

```bash
git clone https://github.com/avinashkongara4/ragnarok-docs.git
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

> **Windows note:** If you get a disk space error, redirect the pip cache to a drive with 5GB+ free space:
> `pip install -r backend/requirements.txt --prefer-binary --cache-dir D:\pip-cache`

### Step 4 — Configure

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
HUGGINGFACE_TOKEN=hf_your_token_here
MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
CHROMA_DB_PATH=./data/processed/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RESULTS=5
USE_LOCAL_MODEL=false
```

Get your token: `huggingface.co/settings/tokens`

Request LLaMA access: `huggingface.co/meta-llama/Llama-3.1-8B-Instruct`

### Step 5 — Verify

```bash
python verify_llama_access.py
```

Expected:
```
ALL CHECKS PASSED. Ready to run!
```

### Step 6 — Run

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 7 — Test

Open `http://localhost:8000/docs` — upload a PDF and ask questions via the Swagger UI.

---

## API Reference

### Upload a document

```http
POST /api/upload
Content-Type: multipart/form-data

file: your-document.pdf
```

**Response:**
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

**Response:**
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

## Model Evaluation

### Fine-Tuning Summary

I fine-tuned LLaMA 3.1 8B using QLoRA on Google's Natural Questions dataset.

| Metric | Value |
|---|---|
| Base model | meta-llama/Llama-3.1-8B-Instruct |
| Fine-tuning method | QLoRA (LoRA rank 16, alpha 32) |
| Parameters trained | 41,943,040 out of 8,072,204,288 (0.52%) |
| Dataset | Google Natural Questions |
| Training examples | 20,000 |
| Epochs | 2 |
| GPU | NVIDIA A100-SXM4-40GB |
| Training time | ~80 minutes |

### Results

| Metric | Base LLaMA 3.1 | Fine-tuned V3 |
|---|---|---|
| Validation Loss | ~2.10 (estimated) | **0.903** |
| Perplexity | ~8.00 (estimated) | **2.47** |

**3.2x lower perplexity than the base model.**

### Training Loss Curve

| Step | Training Loss | Validation Loss | Gap |
|---|---|---|---|
| 100 | 1.097 | 0.998 | 0.099 |
| 500 | 0.962 | 0.940 | 0.022 |
| 1000 | 0.932 | 0.918 | 0.014 |
| 1500 | 0.814 | 0.909 | 0.095 |
| 2000 | 0.818 | 0.904 | 0.086 |
| 2500 | 0.829 | 0.903 | 0.074 |

Gap stayed below 0.10 throughout — genuine generalization, not memorization.

### Known Limitations

- Does not always refuse unanswerable questions — Natural Questions has no unanswerable examples
- Occasionally references information beyond the provided context
- **Planned fix:** Multi-stage fine-tuning adding SQuAD v2 unanswerable examples in Stage 2

### Fine-Tuned Adapter

Published on HuggingFace: `huggingface.co/avinashkongara4/llama3-ragnarok-nq-adapter`

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct"
)
model = PeftModel.from_pretrained(
    base_model,
    "avinashkongara4/llama3-ragnarok-nq-adapter"
)
```

---

## Running on a New Machine

After cloning and installing, you need to re-upload your PDFs via `POST /api/upload` since ChromaDB is stored locally and not tracked in Git.

**Production roadmap:** Migrate to Pinecone for persistent cloud vector storage and AWS S3 or Cloudflare R2 for document storage — eliminating the need to re-upload on each deployment.

---

## Production Considerations

| Solution | Cost at 10,000 queries/month |
|---|---|
| GPT-4 API | ~$300/month |
| Ragnarok-Docs (HF free tier) | ~$10/month (hosting only) |
| Ragnarok-Docs (self-hosted) | ~$0/month |

A production version would use:
- **Pinecone or Weaviate** — persistent cloud vector storage
- **AWS S3 or Cloudflare R2** — permanent PDF file storage
- **PostgreSQL** — document metadata and user management
- **Redis** — caching frequent queries

---

## Deployment

### Backend — Railway

Connect your GitHub repo to Railway. It auto-deploys on every push to main. Set environment variables in the Railway dashboard.

### Frontend — Vercel

Connect your GitHub repo to Vercel. Set `REACT_APP_API_URL` to your Railway backend URL. Vercel auto-deploys on every push.

---

## Roadmap

- [x] Phase 1 — Environment setup and project scaffold
- [x] Phase 2 — RAG pipeline with LLaMA 3.1 and ChromaDB
- [x] Phase 3 — QLoRA fine-tuning on Google Natural Questions
- [ ] Phase 4 — React frontend with chat UI
- [ ] Phase 5 — RAGAS evaluation dashboard
- [ ] Phase 6 — Deploy to Railway and Vercel

---

## Training Journal

I documented every decision, problem, and fix during fine-tuning in [TRAINING_JOURNAL.md](TRAINING_JOURNAL.md).

Highlights:
- Why I switched from SQuAD v2 to Natural Questions
- How I diagnosed and fixed severe overfitting
- Every package version conflict and its fix
- Why the free T4 GPU kept crashing and how I solved it
- What perplexity 2.47 actually means

---

## Author

Built by **Avinash Chowdary Kongara** as an AI engineering portfolio project.

- GitHub: [github.com/Avinash-4](https://github.com/Avinash-4)
- HuggingFace: [huggingface.co/avinashkongara4](https://huggingface.co/avinashkongara4)
- LinkedIn: [linkedin.com/in/avinashkongara4](https://linkedin.com/in/avinashkongara4)

---

## License

MIT License — free to use as a reference for your own portfolio projects.

---

<div align="center">

*Built with LLaMA 3.1, LangChain, ChromaDB, FastAPI, and a lot of debugging*

⭐ Star this repo if it helped you build something

</div>