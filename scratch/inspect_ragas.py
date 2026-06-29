import sys
import os
import inspect

# Allow imports from project root and src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ragas.embeddings.base import BaseRagasEmbeddings

# Get all methods of BaseRagasEmbeddings and print their signatures
print("BaseRagasEmbeddings Abstract methods:")
for name, value in inspect.getmembers(BaseRagasEmbeddings):
    if hasattr(value, "__isabstractmethod__") and value.__isabstractmethod__:
        sig = inspect.signature(value) if callable(value) else ""
        print(f"- {name}{sig}")
