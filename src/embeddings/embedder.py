"""
src/embeddings/embedder.py

Loads the embedding model and converts text into vectors.

This module knows nothing about:
    - PDFs
    - Chunking
    - ChromaDB
    - Retrieval
    - LLMs

It only converts text -> embeddings.
"""

from sentence_transformers import SentenceTransformer

from config import settings


class Embedder:
    def __init__(self) -> None:
        print(
            f"[embedder] Loading model: "
            f"{settings.embedding.model} "
            f"on {settings.embedding.resolved_device}"
        )

        self.model = SentenceTransformer(
            settings.embedding.model,
            device=settings.embedding.resolved_device,
        )

    def encode(self, texts: list[str]):
        """
        Embed multiple texts.

        Returns:
            numpy.ndarray
                shape = (n_texts, embedding_dim)
        """
        return self.model.encode(
            texts,
            batch_size=settings.embedding.batch_size,
            show_progress_bar=True,
            device=settings.embedding.resolved_device,
        )

    def encode_query(self, query: str):
        """
        Embed a single query string.

        Returns:
            numpy.ndarray
                shape = (embedding_dim,)
        """
        return self.model.encode(
            query,
            device=settings.embedding.resolved_device,
        )