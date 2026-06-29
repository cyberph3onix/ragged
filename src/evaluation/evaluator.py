"""
src/evaluation/evaluator.py

Orchestrates running the RAG pipeline on the benchmark dataset and evaluating
the results using RAGAS custom LLM and Embeddings wrappers.
"""

import sys
import os
import asyncio
import pandas as pd
from datasets import Dataset

from config import settings
from llm.provider import LLMProvider
from embeddings.embedder import Embedder
from retrieval.retriever import Retriever
from generation.generator import Generator
from evaluation.dataset import GoldenDataset

from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from langchain_core.outputs import LLMResult, Generation
from langchain_core.prompt_values import PromptValue

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall


class RagasLLMWrapper(BaseRagasLLM):
    """
    Adapter to allow RAGAS to use our LLMProvider (Groq/Ollama/Gemini).
    """
    def __init__(self, provider: LLMProvider):
        super().__init__()
        self.provider = provider

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: list[str] | None = None,
        callbacks = None,
    ) -> LLMResult:
        prompt_str = prompt.to_string()
        response_text = self.provider.generate(prompt_str)
        # Handle n-generations (duplicate to satisfy RAGAS requested shape)
        generations = [Generation(text=response_text) for _ in range(n)]
        return LLMResult(generations=[generations])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: list[str] | None = None,
        callbacks = None,
    ) -> LLMResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_text(prompt, n, temperature, stop, callbacks)
        )

    def is_finished(self, response: LLMResult) -> bool:
        return True


class RagasEmbeddingsWrapper(BaseRagasEmbeddings):
    """
    Adapter to allow RAGAS to use our local Sentence-Transformer Embedder.
    """
    def __init__(self, embedder: Embedder):
        super().__init__()
        self.embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.encode_query(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)


class RagasEvaluator:
    """
    Orchestrates the running of test cases and executing evaluations via Ragas.
    """
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        self.llm_provider = LLMProvider()
        self.embedder = Embedder()

    def run_pipeline_on_dataset(self, golden_ds: GoldenDataset) -> pd.DataFrame:
        """
        Runs the full RAG pipeline (retrieval + generation) on each query in the dataset
        and structures the results as a pandas DataFrame.
        """
        results = []
        for idx, pair in enumerate(golden_ds.pairs, start=1):
            print(f"[{idx}/{len(golden_ds.pairs)}] Processing query: '{pair.question}'")
            
            # 1. Retrieve chunks
            chunks = self.retriever.retrieve(pair.question)
            contexts = [chunk["text"] for chunk in chunks]
            
            # 2. Generate answer
            gen_res = self.generator.generate(pair.question, chunks)
            
            results.append({
                "question": pair.question,
                "contexts": contexts,
                "answer": gen_res["answer"],
                "ground_truth": pair.ground_truth
            })
            
        return pd.DataFrame(results)

    def evaluate_results(self, df: pd.DataFrame) -> dict:
        """
        Executes evaluation on the generated DataFrame using wrapped LLM and embeddings.
        """
        dataset = Dataset.from_pandas(df)
        
        # Instantiate wrappers
        ragas_llm = RagasLLMWrapper(self.llm_provider)
        ragas_embeddings = RagasEmbeddingsWrapper(self.embedder)
        
        # Select metrics and initialize with wrappers
        metrics = [
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ]

        from ragas import evaluate
        
        print("Running RAGAS evaluation framework...")
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        
        return result
