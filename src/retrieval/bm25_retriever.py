"""
src/retrieval/bm25_retriever.py

BM25 keyword retrieval.

Loads a persisted BM25 index and retrieves
the most relevant chunks for a query.
"""

import pickle

from src.config import settings


class BM25Retriever:

    def __init__(self) -> None:

        with open(
            settings.paths.bm25_index,
            "rb",
        ) as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]

    def retrieve(
        self,
        query: str,
    ) -> list[dict]:

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        top_indices = ranked_indices[
            : settings.retrieval.top_k
        ]

        results = []

        for idx in top_indices:

            chunk = self.chunks[idx]

            results.append(
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "chunk_id": chunk["chunk_id"],
                    "retrieval_score": float(scores[idx]),
                }
            )

        return results