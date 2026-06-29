# PROJECT_OVERVIEW.md

# Production-Grade Hybrid RAG System with Evaluation, Observability, and CI/CD

## Overview

This project aims to build a **production-grade Retrieval-Augmented Generation (RAG) system** that goes far beyond the typical "upload PDF and chat" application. The goal is to create a reliable, measurable, and debuggable AI system that follows software engineering best practices and incorporates evaluation, observability, and automated regression testing.

The system combines multiple retrieval techniques, reranking, grounded generation, quality evaluation, tracing, and CI/CD workflows to ensure answer quality and maintainability.

---

# Motivation

Most RAG applications stop after implementing vector search and an LLM. Such systems are difficult to debug, prone to hallucination, and lack mechanisms to measure quality.

This project focuses on solving those problems by introducing:

* Hybrid retrieval
* Cross-encoder reranking
* Citation enforcement
* Automated evaluation
* Observability and tracing
* Regression testing
* CI/CD integration
* Production-ready architecture

The objective is to build an AI system with the same engineering rigor applied to modern software systems.

---

# Goals

The system should:

* Answer questions using external knowledge sources.
* Retrieve highly relevant information using hybrid retrieval.
* Provide grounded responses with source citations.
* Refuse to answer when evidence is insufficient.
* Measure quality continuously.
* Enable debugging through tracing and observability.
* Detect regressions automatically after code changes.
* Be modular and production deployable.

---

# High-Level Architecture

```
Documents
    ↓
Chunking
    ↓
Embedding Generation
    ↓
Vector Database
    ↓
Vector Search
            \
             → Reciprocal Rank Fusion → Cross Encoder Reranker
            /
BM25 Search
    ↓
Top Context
    ↓
LLM
    ↓
Grounded Answer + Citations
```

---

# Major Components

## 1. Ingestion Pipeline

### Supported Documents

* PDF
* TXT
* Markdown
* CSV
* Future:

  * DOCX
  * HTML
  * Databases
  * Websites

### Responsibilities

* Load files
* Extract text
* Preserve metadata
* Generate chunks
* Store source information

### Metadata Example

```python
{
    "source": "attention_is_all_you_need.pdf",
    "page": 8,
    "chunk_id": 143
}
```

---

# 2. Chunking Module

### Strategy

* Chunk size: 800 tokens
* Overlap: 100 tokens

### Goals

* Preserve context
* Avoid information loss
* Improve retrieval quality

### Technologies

* LangChain Text Splitters

---

# 3. Embedding Generation

### Model

```python
BAAI/bge-small-en-v1.5
```

### Responsibilities

Convert chunks into vector representations.

### Technologies

* Sentence Transformers

---

# 4. Vector Database

### Options

#### ChromaDB

Primary vector store.

Future:

* Qdrant
* Pinecone
* Weaviate

### Stores

* Embeddings
* Metadata
* Chunk text

---

# 5. Hybrid Retrieval

Traditional vector search often misses exact keywords.

Hybrid retrieval combines semantic understanding with keyword matching.

---

## Vector Search

Captures:

* Semantic meaning
* Similar concepts
* Natural language intent

---

## BM25 Search

Captures:

* Exact terms
* Names
* Numbers
* Rare keywords

### Library

```python
rank_bm25
```

---

# 6. Reciprocal Rank Fusion (RRF)

Merge vector search results with BM25 results.

Advantages:

* Improved recall
* More robust retrieval
* Better ranking diversity

Pipeline:

```
Vector Search
       \
        → RRF
       /
BM25 Search
```

---

# 7. Cross Encoder Reranker

After RRF, candidate chunks are reranked.

### Model

```python
cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Purpose

Increase precision before sending context to the LLM.

Pipeline:

```
Query
 ↓
Top 20 Documents
 ↓
Cross Encoder
 ↓
Top 5 Documents
```

---

# 8. Grounded Generation

The LLM must answer only using retrieved context.

### Rules

* No external knowledge.
* Cite source chunks.
* Decline uncertain answers.

Example:

```
Self-attention allows tokens to attend to all other tokens [Chunk 12].

Attention complexity is O(n²) [Chunk 17].
```

If insufficient evidence exists:

```
I don't have enough information to answer this question.
```

---

# 9. LLM Layer

### Local Models

Ollama

Examples:

* qwen3:4b
* llama3

### Cloud Models

* Gemini
* Groq
* OpenAI

Future:

Model routing based on cost and latency.

---

# 10. Observability and Monitoring

Understanding failures is essential in production.

---

## Langfuse / LangSmith

Track:

### User Query

```
What is self-attention?
```

### Retrieved Chunks

Top documents selected.

### Reranker Scores

Quality of ranking.

### Prompt

Actual prompt sent to LLM.

### Completion

Generated response.

### Token Usage

Input/output tokens.

### Cost

Cost per request.

### Latency

End-to-end response time.

---

# Metrics

## P50 Latency

Median response time.

## P95 Latency

Worst-case user experience.

## Token Consumption

Track efficiency.

## Cost Per Request

Economic impact.

## Failure Rate

Detect degradation.

---

# 11. Evaluation Framework

Continuous evaluation is critical.

---

## Golden Dataset

Hand-curated benchmark.

Contains:

50–200 verified question-answer pairs.

Example:

```json
{
    "question":
    "Why is transformer complexity O(n²)?",

    "ground_truth":
    "Self-attention compares each token against every other token."
}
```

---

# RAGAS Evaluation

Measure:

## Faithfulness

Are generated answers supported by retrieved context?

---

## Answer Relevancy

Did the system answer the question?

---

## Context Precision

Were retrieved chunks useful?

---

## Context Recall

Did retrieval miss important information?

---

## Semantic Similarity

Generated answer vs ground truth.

---

Example Output

```
Faithfulness       0.94
Answer Relevancy   0.91
Context Precision  0.89
Context Recall     0.92
```

---

# 12. Regression Testing

Every change should be validated.

```
Code Change
     ↓
Run Evaluation
     ↓
Compare Metrics
     ↓
Pass or Fail
```

If quality drops below thresholds:

```
Faithfulness < 0.90

❌ Build Fails
```

This prevents silent degradation.

---

# 13. CI/CD Pipeline

GitHub Actions

Pipeline:

```
Push Code
    ↓
Unit Tests
    ↓
RAG Evaluation
    ↓
Threshold Check
    ↓
Docker Build
    ↓
Deployment
```

---

# Backend

FastAPI

Responsibilities:

* Ingestion API
* Query API
* Evaluation endpoints
* Health checks

---

# Frontend

Options:

### Streamlit

Rapid prototyping.

### Next.js

Production interface.

Features:

* Upload files
* Ask questions
* View citations
* Response streaming
* Conversation history

---

# Containerization

Docker

Services:

* Backend
* Vector Database
* Langfuse
* Frontend

---

# Deployment

Possible Platforms

* Railway
* Render
* VPS
* AWS
* GCP

---

# Project Structure

```
ragged/
│
├── data/
│     ├── pdfs/                      # PDF source files for ingestion
│     ├── bm25_index.pkl             # Serialized BM25 index & chunks
│     └── chroma_db/                 # Persistent ChromaDB vector database
│
├── docs/
│     ├── project_overview.md        # High-level architecture & design
│     └── project_checklist.md       # Implementation tracker & progress
│
├── notebooks/                       # Jupyter notebooks for prototyping
│     ├── 00_simple_local_rag.ipynb
│     ├── phase1.ipynb
│     └── generate_testset.ipynb
│
├── scripts/                         # Command-line utility scripts
│     └── generate_testset.py
│
├── src/                             # Python source code package
│     ├── config.py                  # Global Pydantic Settings system
│     ├── chunking/
│     │     └── chunker.py           # Text chunking logic
│     ├── embeddings/
│     │     └── embedder.py          # SentenceTransformer embeddings
│     ├── generation/
│     │     └── generator.py         # Prompt formulation & generation
│     ├── ingestion/
│     │     └── ingest.py            # Pipeline orchestrator to DB/Index
│     ├── llm/
│     │     ├── __init__.py
│     │     └── provider.py          # Local & Cloud LLM abstractions
│     ├── loaders/
│     │     ├── pdf_loader.py        # PDF text extractor via PyMuPDF
│     │     └── csv_loader.py        # Place for CSV loader (empty)
│     └── retrieval/
│           ├── bm25_retriever.py    # BM25 keyword matching retriever
│           ├── vector_retriever.py  # ChromaDB similarity retriever
│           ├── reranker.py          # Cross-Encoder MiniLM reranking
│           └── retriever.py         # Hybrid Retriever with RRF fusion
│
├── tests/                           # Unit and integration tests (Planned)
├── api/                             # FastAPI REST API (Planned)
├── frontend/                        # Web interface frontend UI (Planned)
├── docker/                          # Dockerfiles & container setups (Planned)
├── github_actions/                  # CI/CD pipeline workflows (Planned)
│
├── main.py                          # CLI entrypoint script
├── test_reranker.py                 # Reranker sanity test script
├── pyproject.toml                   # Project dependencies and configs
├── requirements.txt                 # Exported dependency list
└── uv.lock                          # UV package manager lockfile
```

---

# Future Improvements

## Multi-query Retrieval

Generate multiple search queries automatically.

---

## HyDE Retrieval

Generate hypothetical answers before retrieval.

---

## Query Classification

Classify queries into:

* Factual
* Summarization
* Comparison
* Analytical

---

## Agentic RAG

```
User
 ↓
Planner
 ↓
Retriever
 ↓
Web Search
 ↓
Reranker
 ↓
Answer
```

---

## Memory

Conversation history.

---

## Tool Calling

* Calculator
* Web search
* SQL database access

---

## Multi-modal RAG

Support:

* Images
* Tables
* PDFs with diagrams

---

# Tech Stack

## Backend

* Python
* FastAPI

## Embeddings

* Sentence Transformers
* BGE-small-en-v1.5

## Vector Database

* ChromaDB

## Retrieval

* BM25
* Reciprocal Rank Fusion

## Reranking

* Cross Encoder

## LLM

* Ollama
* Gemini
* Groq

## Evaluation

* RAGAS

## Observability

* Langfuse
* LangSmith

## CI/CD

* GitHub Actions

## Deployment

* Docker
* Railway
* Render

---

# Expected Learning Outcomes

By completing this project, the developer will gain experience in:

* Retrieval systems
* Embedding models
* Vector databases
* Hybrid search
* Reranking
* Prompt engineering
* Grounded generation
* Evaluation methodologies
* Observability and tracing
* CI/CD workflows
* Production AI architecture
* Reliability engineering

---

# End Goal

The objective is not merely to build a chatbot.

The objective is to build a measurable, reliable, debuggable, and production-ready AI system where model outputs are treated with the same engineering rigor as traditional software systems.

This project aims to demonstrate practical understanding of modern RAG architectures and production LLM engineering principles.
