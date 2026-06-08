"""
src/chunking/chunker.py

Splits a list of page dicts into overlapping text chunks.
Exact same logic as the notebook — just wrapped in a function.

Returns:
    list of dicts:
        {
            "text":     str,   # chunk text
            "page":     int,   # source page number (0-indexed)
            "source":   str,   # source filename
            "chunk_id": str,   # unique string ID "0", "1", "2", ...
        }
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: list[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """
    Split page dicts into overlapping chunks using RecursiveCharacterTextSplitter.

    Args:
        pages:         Output of load_pdfs() — list of {text, page, source}.
        chunk_size:    Max characters per chunk (from settings.chunking.chunk_size).
        chunk_overlap: Overlap between chunks (from settings.chunking.chunk_overlap).

    Raises:
        RuntimeError: if no chunks are produced (e.g. all pages were empty).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[dict] = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            chunks.append({
                "text":     chunk_text,
                "page":     page["page"],
                "source":   page["source"],
                "chunk_id": str(len(chunks)),  # global unique ID across all pages
            })

    if not chunks:
        raise RuntimeError("No chunks produced — check that pages contain text.")

    print(f"[chunker] {len(pages)} pages → {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks