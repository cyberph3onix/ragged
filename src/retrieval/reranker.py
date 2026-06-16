"""
src/retrieval/reranker.py

Cross-Encoder reranking for retrieval.

Purpose:
    Improve retrieval precision after hybrid search.

Why reranking?

    Embedding models are optimized for recall:
        "find potentially relevant chunks"

    Cross Encoders are optimized for precision:
        "which chunk best answers this query?"

Unlike embeddings, a Cross Encoder looks at
the query and chunk together and directly
predicts their relevance.

Example:

    Query:
        "Why did the Tin Woodman rust?"

    Chunk A:
        "The Tin Woodman forgot to oil himself
        and was caught in a rainstorm."

    Chunk B:
        "The Tin Woodman wanted a heart."

    The Cross Encoder should rank
    Chunk A higher than Chunk B.

Input:
    query: str
    chunks: list[dict]

Output:
    list[dict]

Each chunk receives:

    {
        ...
        "rerank_score": float
    }

Chunks are returned sorted by
descending rerank_score.
"""



from sentence_transformers import CrossEncoder

from src.config import settings


class Reranker:

    def __init__(self) -> None:

        print(
            f"[reranker] Loading model: "
            f"{settings.reranker.model}"
        )

        self.model = CrossEncoder(
            settings.reranker.model
        )

    def rerank(
        self,
        query: str,
        chunks: list[dict],
    ) -> list[dict]:

        if not chunks:
            return []

        pairs = [
            (query, chunk["text"])
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        chunks.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return chunks