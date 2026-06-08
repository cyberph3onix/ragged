"""
main.py

Entry point. Two modes:

  Ingest (run once to build the DB):
    python main.py --ingest

  Query (ask a question):
    python main.py --query "What is RAG?"

  Both together (ingest then query):
    python main.py --ingest --query "What is RAG?"
"""

import argparse
import sys

# -- make `from config import settings` and `from loaders import ...` work
# when running as `python main.py` from the project root
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import settings
from loaders.pdf_loader import load_pdfs
from chunking.chunker import chunk_pages


# ── Lazy imports for heavy dependencies ──────────────────────────────────────
# Imported inside functions so `python main.py --help` is instant.


def run_ingest() -> None:
    """Load PDFs → chunk → embed → store in ChromaDB."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    print("\n=== INGEST ===")

    # 1. Load
    pages = load_pdfs(settings.paths.pdf_folder)

    # 2. Chunk
    chunks = chunk_pages(
        pages,
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
    )

    # 3. Embed
    print(f"[embedder] Loading model: {settings.embedding.model} on {settings.embedding.resolved_device}")
    model = SentenceTransformer(
        settings.embedding.model,
        device=settings.embedding.resolved_device,
    )
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=settings.embedding.batch_size,
        show_progress_bar=True,
        device=settings.embedding.resolved_device,
    )
    print(f"[embedder] Embedded {len(texts)} chunks → shape {embeddings.shape}")

    # 4. Store in ChromaDB
    settings.paths.chroma_db.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.paths.chroma_db))

    # always start fresh on ingest — safe to re-run
    try:
        client.delete_collection(settings.chroma.collection_name)
    except Exception:
        pass
    collection = client.create_collection(settings.chroma.collection_name)

    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": c["source"], "page": c["page"], "chunk_id": c["chunk_id"]}
            for c in chunks
        ],
    )
    print(f"[chroma] Collection '{settings.chroma.collection_name}' "
          f"now has {collection.count()} chunks")
    print("=== INGEST DONE ===\n")


def run_query(query: str) -> None:
    """Retrieve relevant chunks and generate a cited answer."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    from ollama import chat

    print(f"\n=== QUERY: {query!r} ===")

    # Load embedding model
    model = SentenceTransformer(
        settings.embedding.model,
        device=settings.embedding.resolved_device,
    )

    # Connect to existing ChromaDB
    if not settings.paths.chroma_db.exists():
        print("[error] ChromaDB not found. Run with --ingest first.")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(settings.paths.chroma_db))
    try:
        collection = client.get_collection(settings.chroma.collection_name)
    except Exception:
        print(f"[error] Collection '{settings.chroma.collection_name}' not found. Run --ingest first.")
        sys.exit(1)

    # Retrieve
    query_embedding = model.encode(query, device=settings.embedding.resolved_device)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=settings.retrieval.top_k,
    )

    # Show retrieved chunks
    print(f"\n--- Top {settings.retrieval.top_k} chunks ---")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        print(f"[{i+1}] {meta['source']} p.{meta['page']}  dist={dist:.4f}")
        print(f"     {doc[:120].strip()}...")
    print()

    # Build context and sources string
    context = "\n\n".join(results["documents"][0])
    sources = ", ".join(
        f"{m['source']} (p.{m['page']})" for m in results["metadatas"][0]
    )

    # Generate
    prompt = (
        f"{settings.prompts.system}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Sources available: {sources}"
    )

    try:
        response = chat(
            model=settings.llm.model,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response["message"]["content"]
    except Exception as e:
        print(f"[error] Ollama call failed: {e}")
        print("Make sure Ollama is running:  ollama serve")
        sys.exit(1)

    print("--- Answer ---")
    print(answer)
    print("=== DONE ===\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGGED — local RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ingest
  python main.py --query "What is retrieval augmented generation?"
  python main.py --ingest --query "What is RAG?"
        """,
    )
    parser.add_argument("--ingest", action="store_true",
                        help="Load PDFs, embed, and store in ChromaDB")
    parser.add_argument("--query", type=str, default=None,
                        help="Question to answer using the RAG pipeline")
    args = parser.parse_args()

    if not args.ingest and not args.query:
        parser.print_help()
        sys.exit(0)

    if args.ingest:
        run_ingest()

    if args.query:
        run_query(args.query)


if __name__ == "__main__":
    main()