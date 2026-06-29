import sys
import os

# Allow imports from project root and src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import settings

try:
    from ragas.llms.base import BaseRagasLLM
    print("OK: Successfully imported BaseRagasLLM")
except Exception as e:
    print("FAIL: Failed to import BaseRagasLLM:", e)

try:
    from langchain_core.outputs import LLMResult, Generation
    from langchain_core.prompt_values import PromptValue
    print("OK: Successfully imported LangChain Core objects")
except Exception as e:
    print("FAIL: Failed to import LangChain Core objects:", e)

try:
    from ragas.embeddings.base import BaseRagasEmbeddings
    print("OK: Successfully imported BaseRagasEmbeddings")
except Exception as e:
    print("FAIL: Failed to import BaseRagasEmbeddings:", e)
