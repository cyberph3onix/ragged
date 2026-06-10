"""
src/generation/generator.py

Takes retrieved chunks and a user query,
builds a prompt, calls Ollama, and returns an answer.
"""

from llm.provider import LLMProvider

from config import settings


class Generator:
    def generate(
        self,
        query: str,
        chunks: list[dict],
    ) -> dict:
        """
        Generate an answer using retrieved chunks.

        Returns:
            {
                "answer": str,
                "sources": list[dict]
            }
        """

        if not chunks:
            return {
                "answer": (
                    "I don't have enough information "
                    "in the provided documents to answer this."
                ),
                "sources": [],
            }

        # Build context from retrieved chunks
        context = "\n\n".join(
            chunk["text"]
            for chunk in chunks
        )

        # Build source list for the prompt
        sources = ", ".join(
            f"{chunk['source']} (p.{chunk['page']})"
            for chunk in chunks
        )

        prompt = (
            f"{settings.prompts.system}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Sources available: {sources}"
        )
        print(f"[generator] Using model: {settings.llm.model}")
        llm = LLMProvider()
        response = llm.generate(prompt)

        return {
            "answer": response,
            "sources": chunks,
        }