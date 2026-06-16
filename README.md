# 🚀 RAGGED

> A privacy-first, fully local Retrieval-Augmented Generation (RAG) system built from scratch in Python.

RAGGED transforms your documents into a searchable knowledge base and allows Large Language Models (LLMs) to answer questions grounded in your own data.

Instead of relying solely on an LLM's training data, RAGGED retrieves relevant information from your documents at query time and injects that context into the model's prompt, dramatically improving factual accuracy and reducing hallucinations.

Everything runs locally on your machine.

No OpenAI required. No cloud dependency. No vendor lock-in.

---

# Why RAG?

Large Language Models are powerful, but they have two major limitations:

* They cannot access your private documents.
* They can generate incorrect or hallucinated answers.

Retrieval-Augmented Generation solves this by introducing a retrieval layer.

When a user asks a question:

1. Relevant document chunks are retrieved.
2. Retrieved context is added to the prompt.
3. The LLM generates an answer grounded in those documents.

This allows the model to answer questions about information it was never trained on.

---

# How RAGGED Works

```text
       Documents
           │
           ▼
       PDF Loader
           │
           ▼
        Chunking
       /        \
      ▼          ▼
 Embeddings   BM25 Index
 (Semantic)   (Lexical)
      │          │
      ▼          ▼
  ChromaDB       │
      \          /
       ▼        ▼
     Hybrid Search
    (RRF Fusion)
         │
         ▼
    Cross-Encoder
      Reranking
         │
         ▼
 Prompt Construction
         │
         ▼
     Local LLM
         │
         ▼
      Answer
```

Example query:

```text
Question:
"Why did the Tin Woodman rust?"

↓

Hybrid Search (Vector Similarity + BM25 Lexical)

↓

Reciprocal Rank Fusion (RRF) Candidates Merging

↓

Cross-Encoder Reranking (MS-MARCO MiniLM)

↓

Context Construction (Top Chunks Sorted by Rerank Score)

↓

LLM Generation

↓

Grounded Answer
```

---

# Features

## Document Ingestion

* PDF document loading via PyMuPDF
* Multi-document knowledge bases
* Automatic metadata extraction
* Source tracking

## Chunking

* Recursive character splitting
* Configurable chunk size
* Configurable overlap
* Context preservation

## Embeddings

* BAAI/bge-small-en-v1.5
* GPU acceleration support
* CPU fallback support
* Batch embedding generation

## Vector Storage

* ChromaDB persistence
* Local vector database
* Fast similarity search

## Lexical Indexing
* BM25 retrieval powered by `rank-bm25`
* Persistent lexical index serialization

## Hybrid Retrieval & Reranking
* Hybrid retrieval mode combining Vector (semantic) + BM25 (lexical) search
* Reciprocal Rank Fusion (RRF) for robust scoring combinations
* Cross-Encoder Reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2` to maximize retrieval precision

## Generation

* Ollama support
* Groq support
* Gemini support
* Provider abstraction layer

## Configuration

* Environment variables
* Centralized Settings system (Pydantic Settings)

## Privacy

* Fully local workflow
* No mandatory cloud services
* User-controlled data

---

# Installation

## Prerequisites

* Python 3.12+
* UV
* Ollama (optional for local inference)

Install UV:

```bash
pip install uv
```

Clone the repository:

```bash
git clone <repo-url>
cd ragged
```

Create the environment:

```bash
uv sync
```

---

# Configuration

Create a `.env` file:

```env
RAG_LLM__PROVIDER=groq
RAG_LLM__MODEL=llama-3.3-70b-versatile
RAG_LLM__GROQ_API_KEY=YOUR_API_KEY
```

Or use local Ollama:

```env
RAG_LLM__PROVIDER=ollama
RAG_LLM__MODEL=qwen3:4b
```

---

# Quick Start

## Step 1 — Add Documents

Place PDFs in:

```text
data/pdfs/
```

## Step 2 — Build the Knowledge Base

```bash
python main.py --ingest
```

## Step 3 — Ask Questions

```bash
python main.py --query "What is Retrieval-Augmented Generation?"
```

---

# Current Status

## Phase 1 — Complete
* PDF ingestion
* Recursive chunking
* Embedding generation
* ChromaDB persistence
* Vector retrieval
* Multi-provider LLM support
* End-to-end RAG pipeline

## Phase 2 — Complete
* BM25 Lexical retrieval
* Hybrid retrieval mode
* Reciprocal Rank Fusion (RRF)
* Retrieval diagnostics

## Phase 3 — Complete
* Cross-Encoder reranker integration
* Context precision optimization via MS-MARCO MiniLM
* Hallucination mitigation

---

# Roadmap

## Phase 4 — Evaluation
* Golden datasets
* Faithfulness metrics
* Retrieval evaluation
* Benchmarking

## Phase 5 — User Experience
* Gradio UI
* Drag-and-drop uploads
* Streaming responses
* Chat interface

## Phase 6 — Production
* Docker support
* REST API
* Automated testing
* CI/CD
* Monitoring

---

# Tech Stack

* Python
* ChromaDB
* Sentence Transformers
* Ollama
* Groq
* Gemini
* PyMuPDF
* LangChain Text Splitters
* Pydantic
* UV

---

# Author

**Shreshta Raaj Gupta**

Computer Science Engineering Student • AI Engineer • DevOps Enthusiast

Building systems to understand how they work under the hood—not just how to use them.
