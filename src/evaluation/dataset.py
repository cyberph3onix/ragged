"""
src/evaluation/dataset.py

Handles loading, validating, and saving the Golden Q&A Benchmark Dataset.
"""

from pathlib import Path
import json
from typing import Optional
from pydantic import BaseModel, Field
from config import settings


class GoldenQAPair(BaseModel):
    """
    Represents a single verified Question-Answer pair in the benchmark dataset.
    """
    question: str = Field(..., description="The query to ask the RAG pipeline.")
    ground_truth: str = Field(..., description="The verified ground truth answer.")
    reference_contexts: list[str] = Field(
        default_factory=list,
        description="Exact supporting passages from the source document."
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Source metadata: source PDF, page number, chunk ID."
    )


class GoldenDataset(BaseModel):
    """
    Represents a collection of GoldenQAPair objects.
    """
    pairs: list[GoldenQAPair] = Field(default_factory=list)

    @classmethod
    def load_from_json(cls, path: Path) -> "GoldenDataset":
        """
        Loads the golden dataset from a JSON file.
        Supports both list-of-pairs format and object-wrapped format.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Golden dataset file not found at: {path}\n"
                f"Please run the testset generator script first to generate it."
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            # If the JSON is directly a list of QA pairs
            pairs = [GoldenQAPair(**item) for item in data]
            return cls(pairs=pairs)
        elif isinstance(data, dict) and "pairs" in data:
            # If the JSON is an object with a "pairs" key
            return cls(**data)
        else:
            raise ValueError(
                f"Unsupported JSON structure in golden dataset at {path}. "
                f"Expected list of QA pairs or object containing 'pairs'."
            )

    def save_to_json(self, path: Path) -> None:
        """
        Saves the golden dataset to a JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        # We serialize as a flat list of dicts for simple reading/editing
        data_to_save = [pair.model_dump() for pair in self.pairs]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
