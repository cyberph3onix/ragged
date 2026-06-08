"""
src/config.py

Single source of truth for every setting.
All other modules do:  from config import settings

Load order (later overrides earlier):
  1. config.yaml   — tracked in git, safe defaults
  2. .env          — git-ignored, local overrides
  3. RAG_* env vars — for CI / quick terminal tweaks
      e.g.  RAG_LLM__MODEL=llama3:8b  RAG_RETRIEVAL__TOP_K=10
"""

from __future__ import annotations

import yaml

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None
    _TORCH_AVAILABLE = False

from pathlib import Path
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    chunk_size:    int = 800
    chunk_overlap: int = 100

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
    model:      str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    device:     str = "auto"

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
    candidate_k: int   = 20
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
    model:       str   = "qwen3:4b"
    temperature: float = 0.0
    ollama_host: str   = "http://localhost:11434"


class PromptSettings(BaseSettings):
    system: str = (
        "You are a precise research assistant. "
        "Answer ONLY using the provided context. "
        "If the context does not contain enough information to answer, say exactly: "
        "'I don't have enough information in the provided documents to answer this.' "
        "Do not speculate. Do not use prior knowledge. "
        "After your answer, cite sources as: [filename, page N]."
    )


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
    prompts:   PromptSettings    = PromptSettings()
    eval:      EvalSettings      = EvalSettings()


def _load_yaml() -> dict:
    p = PROJECT_ROOT / "config.yaml"
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _build_settings() -> Settings:
    d = _load_yaml()
    return Settings(
        paths=PathSettings(**d.get("paths", {})),
        chunking=ChunkingSettings(**d.get("chunking", {})),
        embedding=EmbeddingSettings(**d.get("embedding", {})),
        chroma=ChromaSettings(**d.get("chroma", {})),
        retrieval=RetrievalSettings(**d.get("retrieval", {})),
        reranker=RerankerSettings(**d.get("reranker", {})),
        llm=LLMSettings(**d.get("llm", {})),
        prompts=PromptSettings(**d.get("prompts", {})),
        eval=EvalSettings(**d.get("eval", {})),
    )


settings = _build_settings()


# ── Quick sanity check: python src/config.py ─────────────────────────────────
if __name__ == "__main__":
    print("✓ Config loaded\n")
    print(f"  project root   : {PROJECT_ROOT}")
    print(f"  pdf_folder     : {settings.paths.pdf_folder}")
    print(f"  chroma_db      : {settings.paths.chroma_db}")
    print(f"  chunk_size     : {settings.chunking.chunk_size}")
    print(f"  chunk_overlap  : {settings.chunking.chunk_overlap}")
    print(f"  embedding model: {settings.embedding.model}")
    print(f"  device         : {settings.embedding.resolved_device}")
    print(f"  llm model      : {settings.llm.model}")
    print(f"  retrieval mode : {settings.retrieval.mode}")
    print(f"  top_k          : {settings.retrieval.top_k}")