"""
main.py

CLI entry point.

Examples:

    python main.py --ingest

    python main.py --query "What is RAG?"

    python main.py --ingest --query "What is RAG?"
"""

import argparse
import sys
import os

# allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ingestion.ingest import ingest
from retrieval.retriever import Retriever
from generation.generator import Generator


def run_query(query: str) -> None:
    print(f"\n=== QUERY ===")
    print(query)
    print()

    retriever = Retriever()

    chunks = retriever.retrieve(query)

    print(f"Retrieved {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks, start=1):

        metric = (
            f"RRF={chunk.get('rrf_score', 0.0):.4f} | "
            f"Rerank={chunk.get('rerank_score', 0.0):.4f}"
        )

        print(
            f"[{i}] "
            f"{chunk['source']} "
            f"p.{chunk['page']} "
            f"({metric})"
        )

        print(
            chunk["text"][:120]
            .replace("\n", " ")
        )

        print()

    generator = Generator()

    result = generator.generate(
        query=query,
        chunks=chunks,
    )

    print("=== ANSWER ===\n")
    print(result["answer"])

    print("\n=== SOURCES ===")

    seen = set()

    for source in result["sources"]:
        key = (
            source["source"],
            source["page"],
        )

        if key in seen:
            continue

        seen.add(key)

        print(
            f"- {source['source']} "
            f"(page {source['page']})"
        )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGGED - Local RAG Pipeline"
    )

    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Build ChromaDB from PDFs",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Question to ask",
    )

    args = parser.parse_args()

    if not args.ingest and not args.query:
        parser.print_help()
        sys.exit(0)

    if args.ingest:
        ingest()

    if args.query:
        run_query(args.query)


if __name__ == "__main__":
    main()