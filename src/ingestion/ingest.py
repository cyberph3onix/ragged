"""
src/ingestion/ingest.py

Builds the vector database from PDFs.

Pipeline:
    PDFs
      ↓
    Pages
      ↓
    Chunks
      ↓
    Embeddings
      ↓
    ChromaDB
"""

import chromadb
import pickle

from rank_bm25 import BM25Okapi

from config import settings
from loaders.pdf_loader import load_pdfs
from chunking.chunker import chunk_pages
from embeddings.embedder import Embedder


def ingest() -> None:
    print("\n=== INGEST ===")

    # 1. Load PDFs
    pages = load_pdfs(settings.paths.pdf_folder)

    # 2. Chunk pages
    chunks = chunk_pages(
        pages,
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
    )

    # 3. Embed chunks
    embedder = Embedder()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedder.encode(texts)

    # Build BM25 index
    tokenized_chunks = [
    chunk["text"].lower().split()
    for chunk in chunks
]
    bm25 = BM25Okapi(tokenized_chunks)
    print(
        f"[embedder] Embedded {len(texts)} chunks "
        f"→ shape {embeddings.shape}"
    )

    # 4. Create Chroma storage folder
    settings.paths.chroma_db.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save BM25 index
    with open(settings.paths.bm25_index, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunks": chunks,
            },
            f,
        )

    print(
        f"[bm25] Saved index to "
        f"{settings.paths.bm25_index}"
    )

    client = chromadb.PersistentClient(
        path=str(settings.paths.chroma_db)
    )

    # Fresh rebuild every ingest
    try:
        client.delete_collection(
            settings.chroma.collection_name
        )
    except Exception:
        pass

    collection = client.create_collection(
        settings.chroma.collection_name
    )

    # 5. Store everything
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": c["source"],
                "page": c["page"],
                "chunk_id": c["chunk_id"],
            }
            for c in chunks
        ],
    )

    print(
        f"[chroma] Collection "
        f"'{settings.chroma.collection_name}' "
        f"contains {collection.count()} chunks"
    )

    print("=== INGEST DONE ===\n")