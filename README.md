# 🚀 RAGGED

> A privacy-first, fully local Retrieval-Augmented Generation (RAG) system built from scratch in Python.

RAGGED transforms your documents into a searchable knowledge base and allows Large Language Models (LLMs) to answer questions grounded in your own data.

Instead of relying solely on an LLM's training data, RAGGED retrieves relevant information from your documents at query time and injects that context into the model's prompt, dramatically improving factual accuracy and reducing hallucinations.

Everything runs locally on your machine. No OpenAI required. No cloud dependency. No vendor lock-in.

---

## Why RAG?

Large Language Models are powerful, but they have two major limitations:

* They cannot access your private documents.
* They can generate incorrect or hallucinated answers.

Retrieval-Augmented Generation solves this by introducing a retrieval layer. When a user asks a question:

1. Relevant document chunks are retrieved.
2. Retrieved context is added to the prompt.
3. The LLM generates an answer grounded in those documents.

This allows the model to answer questions about information it was never trained on.

---

## How RAGGED Works

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
         │
         ▼
   RAGAS Evaluation
  (Quality Gate)
```

A query flows through hybrid search, fusion, reranking, and generation:

```text
Question: "Why did the Tin Woodman rust?"
   ↓  Hybrid Search        Vector similarity + BM25 lexical
   ↓  Reciprocal Rank Fusion   Merge candidate rankings (RRF)
   ↓  Cross-Encoder Rerank     MS-MARCO MiniLM precision pass
   ↓  Context Construction     Top chunks sorted by rerank score
   ↓  LLM Generation
Grounded Answer
```

---

## Features

### Document Ingestion
* PDF loading via PyMuPDF
* Multi-document knowledge bases
* Automatic metadata extraction and source tracking

### Chunking
* Recursive character splitting
* Configurable chunk size and overlap
* Context preservation

### Embeddings
* `BAAI/bge-base-en-v1.5`
* GPU acceleration with CPU fallback
* Batch embedding generation

### Vector Storage
* ChromaDB persistence
* Local vector database with fast similarity search

### Lexical Indexing
* BM25 retrieval powered by `rank-bm25`
* Persistent lexical index serialization

### Hybrid Retrieval & Reranking
* Hybrid mode combining vector (semantic) + BM25 (lexical) search
* Reciprocal Rank Fusion (RRF) for robust score combination
* Cross-Encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to maximize precision

### Generation
* Ollama, Groq, and Gemini support
* Provider abstraction layer with automatic rate-limit backoff

### Evaluation
* RAGAS-based quality evaluation over a golden dataset
* Faithfulness, answer relevancy, and answer correctness metrics
* Automated LLM-generated benchmark testsets
* Markdown reports with per-question score breakdowns
* Threshold quality gates suitable for CI/CD

### Configuration
* Centralized settings via Pydantic Settings
* Environment-variable and `.env` overrides
* A dedicated evaluation LLM, separate from the generation LLM

### Privacy
* Fully local workflow
* No mandatory cloud services
* User-controlled data

---

## Installation

### Prerequisites

* Python 3.12+
* [UV](https://github.com/astral-sh/uv)
* Ollama (optional, for local inference)

Install UV and set up the environment:

```bash
pip install uv

git clone <repo-url>
cd ragged

uv sync
```

---

## Configuration

Create a `.env` file in the project root. Settings use the `RAG_` prefix with `__` as a nested delimiter.

Use a cloud provider for generation:

```env
RAG_LLM__PROVIDER=groq
RAG_LLM__MODEL=llama-3.3-70b-versatile
RAG_LLM__GROQ_API_KEY=YOUR_API_KEY
```

Or run fully local with Ollama:

```env
RAG_LLM__PROVIDER=ollama
RAG_LLM__MODEL=qwen3:4b
```

The evaluation judge is configured independently so you can pair a small generation model with a stronger judge:

```env
RAG_EVAL_LLM__PROVIDER=groq
RAG_EVAL_LLM__MODEL=openai/gpt-oss-120b
```

> [!NOTE]
> Free-tier providers enforce per-minute and per-day request/token caps. RAGGED backs off and retries on rate limits, but a full evaluation run issues several LLM calls per test case — plan your provider and model choices accordingly.

---

## Quick Start

**1. Add documents** — place PDFs in `data/pdfs/`.

**2. Build the knowledge base:**

```bash
python main.py --ingest
```

**3. Ask questions:**

```bash
python main.py --query "What is Retrieval-Augmented Generation?"
```

Each answer is printed alongside the retrieved chunks (with RRF and rerank scores) and the source pages they came from.

---

## Evaluation

RAGGED ships with a RAGAS-based evaluation harness to measure and guard retrieval and generation quality.

**Generate a benchmark testset** from your PDFs (LLM-authored question/answer pairs):

```bash
python scripts/generate_testset.py --num-chunks 10
```

**Run the evaluation and quality gate:**

```bash
python main.py --eval
# or, with more control:
python scripts/run_eval.py --num-samples 10 --metrics faithfulness answer_relevancy answer_correctness
```

The runner executes the full RAG pipeline over the golden dataset, scores each answer with RAGAS, writes a timestamped Markdown report to `data/eval_reports/`, and exits non-zero if any gated metric falls below its threshold — ready to drop into CI.

### Latest Benchmark

Evaluated on a 10-case golden dataset (hybrid retrieval + Cross-Encoder reranking, generation via `openai/gpt-oss-20b`, judged by `openai/gpt-oss-120b`):

| Metric | Score | Threshold | Status |
| --- | --- | --- | --- |
| Faithfulness | 0.92 | 0.75 | ✅ Pass |
| Answer Relevancy | 0.85 | 0.70 | ✅ Pass |
| Answer Correctness | 0.65 | — | — |

Faithfulness measures how well answers are grounded in the retrieved context, answer relevancy how directly they address the question, and answer correctness how closely they match the ground-truth answer.

Useful flags:

| Flag | Description |
| --- | --- |
| `--num-samples N` | Evaluate only the first `N` cases (quick smoke test) |
| `--provider` / `--model` | Override the generation LLM |
| `--eval-provider` / `--eval-model` | Override the RAGAS judge LLM |
| `--metrics` | Choose metrics: `faithfulness`, `answer_relevancy`, `answer_correctness`, `context_precision`, `context_recall` |

---

## Project Status

| Phase | Scope | Status |
| --- | --- | --- |
| **Phase 1** | PDF ingestion, chunking, embeddings, ChromaDB, vector retrieval, multi-provider LLM, end-to-end pipeline | ✅ Complete |
| **Phase 2** | BM25 lexical retrieval, hybrid mode, Reciprocal Rank Fusion, retrieval diagnostics | ✅ Complete |
| **Phase 3** | Cross-Encoder reranking, context-precision optimization, hallucination mitigation | ✅ Complete |
| **Phase 4** | Golden datasets, RAGAS metrics, quality gates, benchmark reporting | ✅ Complete |

### Roadmap

* **Phase 5 — User Experience:** Gradio UI, drag-and-drop uploads, streaming responses, chat interface
* **Phase 6 — Production:** Docker support, REST API, automated testing, CI/CD, monitoring

---

## Tech Stack

* Python
* ChromaDB
* Sentence Transformers
* RAGAS
* Ollama · Groq · Gemini
* PyMuPDF
* LangChain Text Splitters
* Pydantic
* UV

---

## Author

**Shreshta Raaj Gupta**

Computer Science Engineering Student • AI Engineer • DevOps Enthusiast

Building systems to understand how they work under the hood—not just how to use them.
