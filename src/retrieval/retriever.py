"""
src/retrieval/retriever.py

Hybrid retrieval orchestrator.

Currently:
    vector retrieval
    +
    BM25 retrieval

Future:
    vector
    + bm25
    + RRF
    + reranker
"""

from retrieval.vector_retriever import Retriever as VectorRetriever
from retrieval.bm25_retriever import BM25Retriever
from config import settings


class Retriever:

    def __init__(self) -> None:
        self.vector = VectorRetriever()
        self.bm25 = BM25Retriever()

    def retrieve(
        self,
        query: str,
    ) -> list[dict]:

        vector_results = self.vector.retrieve(query)

        bm25_results = self.bm25.retrieve(query)

        rrf_scores = {}

        k = 60

        # Vector rankings
        for rank, chunk in enumerate(vector_results, start=1):

            chunk_id = chunk["chunk_id"]

            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    "chunk": chunk,
                    "score": 0.0,
                }

            rrf_scores[chunk_id]["score"] += (
                1 / (k + rank)
            )

        # BM25 rankings
        for rank, chunk in enumerate(bm25_results, start=1):

            chunk_id = chunk["chunk_id"]

            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    "chunk": chunk,
                    "score": 0.0,
                }

            rrf_scores[chunk_id]["score"] += (
                1 / (k + rank)
            )

        ranked = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        return [
            item["chunk"]
            for item in ranked[
                : settings.retrieval.top_k
            ]
        ]