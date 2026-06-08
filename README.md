# 🚀 RAGGED

> A fully local, privacy-first Retrieval-Augmented Generation (RAG) system built from scratch in Python.

RAGGED ingests PDF documents, converts them into embeddings, stores them in a vector database, retrieves relevant context for user queries, and generates answers using a local Large Language Model (LLM).

No OpenAI API. No cloud dependency. No document uploads.

Everything runs locally on your machine.

---

## ✨ Features

### 📄 PDF Ingestion

* Load and process multiple PDF documents
* Extract text page-by-page using PyMuPDF
* Support for building knowledge bases from multiple documents

### ✂️ Intelligent Chunking

* Recursive character-based chunking
* Configurable chunk size and overlap
* Preserves context across chunk boundaries

### 🧠 Semantic Embeddings

* Powered by `BAAI/bge-small-en-v1.5`
* Converts document chunks into dense vector representations
* Supports CPU and GPU acceleration

### 🗄️ Vector Database

* Persistent ChromaDB storage
* Fast semantic similarity search
* Rebuildable and reproducible indexing pipeline

### 🔍 Semantic Retrieval

* Retrieves the most relevant chunks for a query
* Configurable Top-K retrieval
* Source-aware document search

### 🤖 Local LLM Generation

* Powered by Ollama
* Completely offline inference
* Context-grounded answer generation

### ⚙️ Configuration System

* YAML configuration support
* Environment variable overrides
* Centralized settings management

### 🔒 Privacy First

* No external APIs
* No cloud processing
* No data leaves your machine

---

## 🏗️ Architecture

```text
PDF Documents
      │
      ▼
 PDF Loader
      │
      ▼
   Chunker
      │
      ▼
  Embedder
      │
      ▼
  ChromaDB
      │
      ▼
 Retriever
      │
      ▼
 Generator
      │
      ▼
  Response
```

---

## 📂 Project Structure

```text
ragged/
│
├── main.py
│
└── src/
    ├── config.py
    │
    ├── loaders/
    │   └── pdf_loader.py
    │
    ├── chunking/
    │   └── chunker.py
    │
    ├── embeddings/
    │   └── embedder.py
    │
    ├── ingestion/
    │   └── ingest.py
    │
    ├── retrieval/
    │   └── retriever.py
    │
    └── generation/
        └── generator.py
```

---

## ⚡ GPU Acceleration

RAGGED automatically detects available hardware and uses the best available backend.

Supported devices:

* NVIDIA CUDA
* Apple Silicon (MPS)
* CPU fallback

GPU users can expect significantly faster embedding generation and ingestion times.

---

## 🚀 Getting Started

### 1. Ingest Documents

Place your PDFs inside:

```text
data/pdfs/
```

Then build the vector database:

```bash
python main.py --ingest
```

Example:

```text
=== INGEST ===
[pdf_loader] Loaded 21 pages from 2 PDFs
[chunker] 21 pages → 158 chunks
[embedder] Embedded 158 chunks
[chroma] Collection 'documents' now has 158 chunks
=== INGEST DONE ===
```

---

### 2. Query the Knowledge Base

```bash
python main.py --query "What is Retrieval-Augmented Generation?"
```

Example Flow:

```text
Question
    ↓
Semantic Search
    ↓
Top-K Retrieval
    ↓
Context Construction
    ↓
Local LLM Generation
    ↓
Answer
```

---

## 🛠️ Tech Stack

### Core Technologies

* Python
* Ollama
* ChromaDB
* Sentence Transformers

### Models

* `BAAI/bge-small-en-v1.5` (Embeddings)
* `Qwen3` (Generation)

### Libraries

* PyMuPDF
* LangChain Text Splitters
* Pydantic
* PyYAML

---

## 📈 Current Status

### ✅ Prototype (Phase 1 Complete)

Implemented:

* PDF ingestion
* Recursive chunking
* Semantic embeddings
* ChromaDB persistence
* Vector retrieval
* Ollama integration
* End-to-end local RAG pipeline
* Config-driven architecture

The system is fully functional and capable of answering questions using local documents.

---

## 🔮 Coming Soon

### Phase 2 — Hybrid Retrieval

* BM25 Retrieval
* Hybrid Search (BM25 + Vector Search)
* Reciprocal Rank Fusion (RRF)
* Retrieval diagnostics

### Phase 3 — Better Answers

* Cross-Encoder Reranking
* Improved source attribution
* Context compression
* Reduced hallucinations

### Phase 4 — Evaluation Framework

* Golden dataset generation
* Faithfulness scoring
* Answer relevancy evaluation
* Retrieval benchmarking

### Phase 5 — User Interface

* Gradio web application
* Drag-and-drop PDF upload
* Chat-style interface
* Real-time document indexing

### Phase 6 — Production Readiness

* Docker support
* REST API
* Automated testing
* Structured logging
* Monitoring and observability

---

## 🎯 Goals

RAGGED is not just a RAG application.

The goal of this project is to understand, implement, and optimize every major component of a modern Retrieval-Augmented Generation pipeline from scratch while keeping the entire stack local, transparent, and extensible.

---

## 👨‍💻 Author

**Shreshta Raaj Gupta**

Computer Science Engineering Student • AI Engineer • DevOps Enthusiast

Building systems to understand how they work under the hood—not just how to use them.

---

### ⭐ If you find this project interesting, consider starring the repository.
