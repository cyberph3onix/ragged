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
from ragas.run_config import RunConfig
from langchain_core.outputs import LLMResult, Generation
from langchain_core.prompt_values import PromptValue

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._answer_correctness import AnswerCorrectness


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
        # AnswerCorrectness -> AnswerSimilarity reads embeddings.run_config;
        # give it a default and honor set_run_config() from evaluate().
        self.run_config = RunConfig()

    def set_run_config(self, run_config: RunConfig) -> None:
        self.run_config = run_config

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
        self.eval_llm = LLMProvider(
            provider=settings.eval_llm.provider,
            model=settings.eval_llm.model,
        )
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

    def evaluate_results(self, df: pd.DataFrame, metric_names: list[str] | None = None) -> dict:
        """
        Executes evaluation on the generated DataFrame using wrapped LLM and embeddings.

        Args:
            metric_names: Which RAGAS metrics to compute. Defaults to all four.
                For fast iteration pass just the gated metrics
                (["faithfulness", "answer_relevancy"]) to avoid the expensive
                per-context ContextPrecision fan-out.
        """
        dataset = Dataset.from_pandas(df)

        # Instantiate wrappers — use the eval-specific LLM for RAGAS metrics
        ragas_llm = RagasLLMWrapper(self.eval_llm)
        ragas_embeddings = RagasEmbeddingsWrapper(self.embedder)

        # Metric registry — built lazily so we only construct the ones requested.
        metric_factory = {
            "faithfulness":       lambda: Faithfulness(llm=ragas_llm),
            "answer_relevancy":   lambda: AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            "context_precision":  lambda: ContextPrecision(llm=ragas_llm),
            "context_recall":     lambda: ContextRecall(llm=ragas_llm),
            # Correctness compares the answer to ground_truth (not just its
            # phrasing), so short factual answers aren't punished the way
            # answer_relevancy punishes them.
            "answer_correctness": lambda: AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
        }

        if metric_names is None:
            metric_names = list(metric_factory.keys())

        unknown = [m for m in metric_names if m not in metric_factory]
        if unknown:
            raise ValueError(
                f"Unknown metric(s): {unknown}. Valid: {list(metric_factory.keys())}"
            )

        metrics = [metric_factory[name]() for name in metric_names]

        from ragas import evaluate

        # Map our DataFrame columns to RAGAS's expected internal names
        column_map = {
            "user_input": "question",
            "response": "answer",
            "retrieved_contexts": "contexts",
            "reference": "ground_truth",
        }

        # Local Ollama is single-threaded on CPU, so keep it sequential. Groq's
        # free tier caps at 12k TPM and context_precision/context_recall send
        # large prompts, so it also runs sequential — the per-call delay in
        # LLMProvider._groq() paces requests to stay under the cap. Gemini
        # tolerates concurrency, which turns dozens of sequential round-trips
        # into a handful of parallel batches.
        # (This gates on eval_llm, not llm — RAGAS calls the *evaluation* LLM.)
        provider = settings.eval_llm.provider.lower()
        max_workers = 1 if provider in ("ollama", "groq") else 2

        # Allow up to 15 min per LLM call and cap retries to avoid spending hours
        # on transient failures.
        run_config = RunConfig(timeout=900, max_retries=3, max_wait=120, max_workers=max_workers)

        print("Running RAGAS evaluation framework...")
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            column_map=column_map,
            run_config=run_config,
            raise_exceptions=True,
        )

        return result
