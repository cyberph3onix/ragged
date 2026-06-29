"""
scripts/generate_testset.py

Automated Benchmark Dataset Generator Script.
Loads pages from PDFs, samples a subset of pages, and queries the LLM (Groq)
to generate high-quality question-answer pairs based on those pages.
"""

import sys
import os
import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm

# Reconfigure stdout/stderr to UTF-8 to prevent encoding errors on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# Allow imports from project root and src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import settings
from loaders.pdf_loader import load_pdfs
from chunking.chunker import chunk_pages
from llm.provider import LLMProvider
from evaluation.dataset import GoldenQAPair, GoldenDataset


def generate_benchmark(num_chunks: int, sample_seed: int = 42) -> None:
    pdf_folder = settings.paths.pdf_folder
    print(f"Loading PDFs from: {pdf_folder}")
    
    try:
        pages = load_pdfs(pdf_folder)
    except Exception as e:
        print(f"Error loading PDFs: {e}")
        sys.exit(1)
        
    total_pages = len(pages)
    if total_pages == 0:
        print("No readable PDF pages found. Please add PDFs to the data/pdfs directory.")
        sys.exit(1)
        
    print(f"Total pages loaded: {total_pages}")
    
    # Chunk pages
    print("Chunking pages...")
    try:
        chunks = chunk_pages(
            pages,
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
        )
    except Exception as e:
        print(f"Error chunking pages: {e}")
        sys.exit(1)
        
    # Filter chunks that are too short/empty
    valid_chunks = [c for c in chunks if len(c.get("text", "").strip()) >= 200]
    total_chunks = len(valid_chunks)
    if total_chunks == 0:
        print("No valid chunks of sufficient length (>= 200 chars) found.")
        sys.exit(1)
        
    # Select chunks to process
    if num_chunks >= total_chunks:
        selected_chunks = valid_chunks
        print(f"Selecting all {total_chunks} chunks for generation.")
    else:
        random.seed(sample_seed)
        selected_chunks = random.sample(valid_chunks, num_chunks)
        print(f"Sampled {num_chunks} chunks out of {total_chunks} (seed: {sample_seed}).")

    llm = LLMProvider()
    dataset_pairs = []
    seen_questions = set()

    print(f"Generating questions using LLM provider: {settings.llm.provider} (Model: {settings.llm.model})")
    
    for idx, chunk in enumerate(tqdm(selected_chunks, desc="Generating QA Pairs")):
        text = chunk.get("text", "").strip()
        
        prompt = (
            "You are a professional benchmark generator. Given the following text chunk, "
            "generate exactly 2 distinct, high-quality question-answer samples based strictly on the text provided. "
            "Do not use any outside knowledge.\n\n"
            "For each sample, you must generate:\n"
            "1. A realistic user question that can be answered using only the provided text chunk.\n"
            "2. A complete and detailed answer (ground truth) to the question, based strictly on the text.\n"
            "3. The exact supporting context (verbatim sentence or passage from the chunk) that contains the information used to answer the question.\n\n"
            "Format the response strictly as a JSON array of objects. Each object must contain exactly these keys:\n"
            "- 'question': a string representing the realistic user question.\n"
            "- 'ground_truth': a string representing the complete and detailed answer.\n"
            "- 'reference_context': a string containing the exact supporting context copied verbatim from the source chunk.\n\n"
            "Important: To ensure valid JSON formatting, do NOT include raw, unescaped double quotes inside your JSON string values. "
            "For the 'reference_context' value (and any other keys), copy the verbatim text but replace any internal double quotes (\") with single quotes (') or escape them properly as \\\" to keep the JSON syntax valid.\n\n"
            "Do not include any preambles, explanations, markdown formatting (do not wrap in ```json), or trailing text.\n\n"
            f"Context from document '{chunk.get('source')}' (page {chunk.get('page') + 1}):\n"
            f"--------------------------------------------------\n"
            f"{text}\n"
            f"--------------------------------------------------\n\n"
            "JSON Output format:\n"
            "[\n"
            "  {\n"
            "    \"question\": \"Realistic user question here?\",\n"
            "    \"ground_truth\": \"Detailed, complete, and accurate answer here.\",\n"
            "    \"reference_context\": \"Exact verbatim supporting sentence or passage from the chunk here.\"\n"
            "  },\n"
            "  {\n"
            "    \"question\": \"Another realistic user question here?\",\n"
            "    \"ground_truth\": \"Another detailed, complete, and accurate answer here.\",\n"
            "    \"reference_context\": \"Another exact verbatim supporting sentence or passage from the chunk here.\"\n"
            "  }\n"
            "]"
        )
        
        try:
            response = llm.generate(prompt).strip()
            
            # Clean markdown codeblocks if model returned them
            if "```json" in response:
                response_clean = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response_clean = response.split("```")[1].split("```")[0].strip()
            else:
                response_clean = response.strip()
                
            qa_list = json.loads(response_clean)
            
            for item in qa_list:
                question = item.get("question", "").strip()
                ground_truth = item.get("ground_truth", "").strip()
                
                # Check for reference context
                ref_ctx = item.get("reference_context") or item.get("reference_contexts")
                if isinstance(ref_ctx, list):
                    reference_contexts = [str(c).strip() for c in ref_ctx if str(c).strip()]
                elif isinstance(ref_ctx, str) and ref_ctx.strip():
                    reference_contexts = [ref_ctx.strip()]
                else:
                    reference_contexts = [text]

                if question and ground_truth:
                    # Deduplicate questions: Normalize the question (lowercase + stripped whitespace)
                    normalized_q = question.lower().strip()
                    if normalized_q in seen_questions:
                        continue
                    seen_questions.add(normalized_q)

                    # Build metadata
                    metadata = {
                        "source": chunk.get("source"),
                        "page": chunk.get("page"),
                    }
                    if "chunk_id" in chunk:
                        metadata["chunk_id"] = chunk["chunk_id"]

                    pair = GoldenQAPair(
                        question=question,
                        ground_truth=ground_truth,
                        reference_contexts=reference_contexts,
                        metadata=metadata
                    )
                    dataset_pairs.append(pair)
                    
        except json.JSONDecodeError as jde:
            print(f"\n[Warning] JSON parse failed for chunk {chunk.get('chunk_id')} in {chunk.get('source')} p.{chunk.get('page') + 1}.")
            print(f"Response was: {response}")
        except Exception as e:
            print(f"\n[Error] Failed generating for chunk {chunk.get('chunk_id')} in {chunk.get('source')} p.{chunk.get('page') + 1}: {e}")

    # Build dataset and save
    golden_dataset = GoldenDataset(pairs=dataset_pairs)
    output_path = settings.paths.golden_dataset
    
    print(f"\nSaving {len(dataset_pairs)} QA pairs to {output_path}...")
    golden_dataset.save_to_json(output_path)
    print("Dataset saved successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGGED - Benchmark Testset Generator"
    )
    
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=10,
        help="Number of random chunks to sample from PDFs for QA generation (default: 10)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random chunk sampling (default: 42)",
    )
    
    args = parser.parse_args()
    generate_benchmark(num_chunks=args.num_chunks, sample_seed=args.seed)


if __name__ == "__main__":
    main()