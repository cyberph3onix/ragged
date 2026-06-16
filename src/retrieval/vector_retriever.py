"""
src/retrieval/retriever.py

Vector retrieval from ChromaDB.

Input:
    user query

Output:
    list of retrieved chunk dicts
"""

import chromadb

from config import settings
from embeddings.embedder import Embedder


class Retriever:
    def __init__(self) -> None:
        self.embedder = Embedder()

        self.client = chromadb.PersistentClient(
            path=str(settings.paths.chroma_db)
        )

        self.collection = self.client.get_collection(
            settings.chroma.collection_name
        )

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve top-k chunks from Chroma.

        Returns:
            [
                {
                    "text": "...",
                    "source": "...",
                    "page": 1,
                    "chunk_id": "123",
                    "retrieval_score": 0.42,
                }
            ]
        """

        query_embedding = self.embedder.encode_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=settings.retrieval.candidate_k,
        )

        retrieved_chunks = []

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved_chunks.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "page": meta["page"],
                    "chunk_id": meta["chunk_id"],
                    "retrieval_score": dist,
                }
            )

        return retrieved_chunks