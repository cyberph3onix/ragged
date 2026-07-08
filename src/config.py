"""
src/config.py

Single source of truth for every setting.
All other modules do:  from config import settings

Load order (later overrides earlier):
  1. .env          — git-ignored, local overrides
  2. RAG_* env vars — for CI / quick terminal tweaks
      e.g.  RAG_LLM__MODEL=llama3:8b  RAG_RETRIEVAL__TOP_K=10
"""

from __future__ import annotations

from pathlib import Path
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None
    _TORCH_AVAILABLE = False


# ── Project root ──────────────────────────────────────────────────────────────
def _find_project_root() -> Path:
    """Walk up from this file until pyproject.toml is found."""
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


PROJECT_ROOT = _find_project_root()


# ── Sub-models ────────────────────────────────────────────────────────────────

class PathSettings(BaseSettings):
    pdf_folder:     Path = Path("data/pdfs")
    chroma_db:      Path = Path("data/chroma_db")
    bm25_index:     Path = Path("data/bm25_index.pkl")
    golden_dataset: Path = Path("data/golden_dataset.json")

    @model_validator(mode="after")
    def _make_absolute(self) -> "PathSettings":
        self.pdf_folder     = PROJECT_ROOT / self.pdf_folder
        self.chroma_db      = PROJECT_ROOT / self.chroma_db
        self.bm25_index     = PROJECT_ROOT / self.bm25_index
        self.golden_dataset = PROJECT_ROOT / self.golden_dataset
        return self


class ChunkingSettings(BaseSettings):
    chunk_size:    int = 400
    chunk_overlap: int = 80

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_less_than_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size", 800)
        if v >= size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({size})"
            )
        return v


class EmbeddingSettings(BaseSettings):
    model:      str = "BAAI/bge-base-en-v1.5"
    batch_size: int = 32
    device:     str = "cpu"  # or "auto", "cuda", "mps"

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        if not _TORCH_AVAILABLE:
            return "cpu"
        if _torch.cuda.is_available():
            return "cuda"
        if _torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class ChromaSettings(BaseSettings):
    collection_name: str = "documents"


class RetrievalSettings(BaseSettings):
    top_k:       int   = 5
    candidate_k: int   = 30
    mode:        str   = "hybrid"
    bm25_weight: float = 0.4

    @field_validator("mode")
    @classmethod
    def _valid_mode(cls, v: str) -> str:
        allowed = {"hybrid", "vector", "bm25"}
        if v not in allowed:
            raise ValueError(f"retrieval.mode must be one of {allowed}, got '{v}'")
        return v


class RerankerSettings(BaseSettings):
    enabled: bool = True
    model:   str  = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LLMSettings(BaseSettings):
    provider: str = "ollama"

    model: str = "qwen3:4b"

    temperature: float = 0.0

    ollama_host: str = "http://localhost:11434"

    groq_api_key: str = ""

    gemini_api_key: str = ""


class PromptSettings(BaseSettings):
    system: str = (
        "You are a precise research assistant. "
        "Answer ONLY using the provided context. Do not speculate or use prior knowledge. "
        "Give a single, direct sentence that answers the question — no preamble, no "
        "meta-commentary, no reasoning about the text, and no hedging. "
        "If, and ONLY if, the context contains no answer at all, reply with exactly: "
        "'I don't have enough information in the provided documents to answer this.' "
        "Never append that sentence to an answer you have already given. "
        "Do not invent chapter numbers, page numbers, or source names that are not "
        "explicitly given to you — only reference the sources listed under "
        "'Sources available' if you need to cite one. "
    )


class EvalLLMSettings(BaseSettings):
    provider: str = "gemini"
    # Flash-Lite gives 1,000 req/day + 15 req/min on the free tier (vs. 20/day
    # for gemini-2.5-flash) and is capable enough for RAGAS's structured JSON.
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.0


class EvalSettings(BaseSettings):
    faithfulness_threshold:     float = 0.75
    answer_relevancy_threshold: float = 0.70


# ── Root settings ─────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    paths:     PathSettings      = PathSettings()
    chunking:  ChunkingSettings  = ChunkingSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    chroma:    ChromaSettings    = ChromaSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    reranker:  RerankerSettings  = RerankerSettings()
    llm:       LLMSettings       = LLMSettings()
    eval_llm:  EvalLLMSettings   = EvalLLMSettings()
    prompts:   PromptSettings    = PromptSettings()
    eval:      EvalSettings      = EvalSettings()


def _build_settings() -> Settings:
    return Settings()

settings = _build_settings()


# ── Quick sanity check: python src/config.py ─────────────────────────────────
if __name__ == "__main__":
    print("✓ Config loaded\n")
    print(f"  project root   : {PROJECT_ROOT}")
    print(f"  pdf_folder     : {settings.paths.pdf_folder}")
    print(f"  chroma_db      : {settings.paths.chroma_db}")
    print(f"  chunk_size      : {settings.chunking.chunk_size}")
    print(f"  chunk_overlap   : {settings.chunking.chunk_overlap}")
    print(f"  embedding model : {settings.embedding.model}")
    print(f"  device          : {settings.embedding.resolved_device}")
    print(f"  gen llm model   : {settings.llm.model}")
    print(f"  eval llm model  : {settings.eval_llm.model}")
    print(f"  retrieval mode  : {settings.retrieval.mode}")
    print(f"  top_k           : {settings.retrieval.top_k}")
    print(f"  candidate_k     : {settings.retrieval.candidate_k}")