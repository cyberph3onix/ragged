import sys
import os
import pandas as pd
from datasets import Dataset

# Allow imports from project root and src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

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

class RagasLLMWrapper(BaseRagasLLM):
    def __init__(self, provider: LLMProvider):
        super().__init__()
        self.provider = provider
        
    def generate_text(
        self,
        prompt: PromptValue,
        n=1,
        temperature=0.01,
        stop=None,
        callbacks=None,
    ) -> LLMResult:
        prompt_str = prompt.to_string()
        response_text = self.provider.generate(prompt_str)
        generation = Generation(text=response_text)
        return LLMResult(generations=[[generation]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n=1,
        temperature=0.01,
        stop=None,
        callbacks=None,
    ) -> LLMResult:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_text(prompt, n, temperature, stop, callbacks)
        )

    def is_finished(self, response: LLMResult) -> bool:
        return True

class RagasEmbeddingsWrapper(BaseRagasEmbeddings):
    def __init__(self, embedder: Embedder):
        super().__init__()
        self.embedder = embedder
        
    def embed_query(self, text: str) -> list[float]:
        return self.embedder.encode_query(text).tolist()
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.embed_query,
            text
        )

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.embed_documents,
            texts
        )

def main():
    print("Loading retriever & generator...")
    retriever = Retriever()
    generator = Generator()
    
    print("Loading golden dataset...")
    ds_path = settings.paths.golden_dataset
    golden_ds = GoldenDataset.load_from_json(ds_path)
    
    # Run evaluation on first 2 pairs to verify
    test_pairs = golden_ds.pairs[:2]
    
    print(f"Running pipeline on {len(test_pairs)} test pairs...")
    eval_data = []
    for pair in test_pairs:
        print(f"Query: {pair.question}")
        chunks = retriever.retrieve(pair.question)
        contexts = [c["text"] for c in chunks]
        res = generator.generate(pair.question, chunks)
        eval_data.append({
            "question": pair.question,
            "contexts": contexts,
            "answer": res["answer"],
            "ground_truth": pair.ground_truth
        })
        
    df = pd.DataFrame(eval_data)
    dataset = Dataset.from_pandas(df)
    
    print("Initializing wrappers...")
    llm_prov = LLMProvider()
    embed_prov = Embedder()
    
    ragas_llm = RagasLLMWrapper(llm_prov)
    ragas_embeddings = RagasEmbeddingsWrapper(embed_prov)
    
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    
    print("Running Ragas evaluate...")
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    
    # Bind LLM and Embeddings to metrics in Ragas 0.4.x
    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings
            
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )
    
    print("\n=== EVALUATION SUCCESS ===")
    print(result)

if __name__ == "__main__":
    main()
