"""
src/loaders/pdf_loader.py

Loads every PDF in a folder and returns a flat list of page dicts.
Exact same logic as the notebook — just wrapped in a function.

Returns:
    list of dicts:
        {
            "text":   str,   # raw page text, stripped
            "page":   int,   # 0-indexed page number
            "source": str,   # filename e.g. "rag_survey.pdf"
        }
"""

import fitz  # PyMuPDF
from pathlib import Path


def load_pdfs(pdf_folder: Path) -> list[dict]:
    """
    Load all PDFs in pdf_folder, return one dict per non-empty page.

    Raises:
        FileNotFoundError: if pdf_folder does not exist.
        RuntimeError:      if no readable pages are found across all PDFs.
    """
    if not pdf_folder.exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")

    pages: list[dict] = []

    for pdf_path in sorted(pdf_folder.glob("*.pdf")):
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text().strip()
                    if text:  # skip blank / image-only pages
                        pages.append({
                            "text":   text,
                            "page":   page_num, 
                            "source": pdf_path.name,
                        })
        except Exception as e:
            print(f"[pdf_loader] Skipping {pdf_path.name}: {e}")

    if not pages:
        raise RuntimeError(f"No readable PDF pages found in {pdf_folder}")

    print(f"[pdf_loader] Loaded {len(pages)} pages from {len(list(pdf_folder.glob('*.pdf')))} PDFs")
    return pages