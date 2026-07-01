"""
scripts/run_eval.py

CLI runner to execute evaluation on the golden dataset, calculate scores,
generate markdown reports, and enforce target threshold gates for CI/CD.
"""

import sys
import os
import argparse
from pathlib import Path

# Allow imports from project root and src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import settings
from evaluation.dataset import GoldenDataset
from evaluation.evaluator import RagasEvaluator
from evaluation.report import generate_markdown_report, extract_summary_scores


def run_evaluation(
    num_samples: int | None = None,
    provider: str = "groq",
    model: str = "llama-3.1-8b-instant",
    metrics: list[str] | None = None,
) -> None:
    # Route both the RAG generation step and the RAGAS eval LLM through the
    # chosen provider. The global `settings` singleton is read at call-time by
    # LLMProvider/Generator, so overriding it here is enough.
    settings.llm.provider = provider
    settings.llm.model = model
    print(f"Evaluation LLM provider: {provider} (model: {model})")

    dataset_path = settings.paths.golden_dataset
    print(f"Loading benchmark dataset from: {dataset_path}")

    try:
        golden_ds = GoldenDataset.load_from_json(dataset_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if num_samples is not None:
        golden_ds.pairs = golden_ds.pairs[:num_samples]

    num_pairs = len(golden_ds.pairs)
    if num_pairs == 0:
        print("Error: The benchmark dataset is empty.")
        sys.exit(1)

    print(f"Loaded {num_pairs} test cases.")
    
    evaluator = RagasEvaluator()
    
    print("\n--- STEP 1: Running RAG Pipeline on Test Cases ---")
    raw_df = evaluator.run_pipeline_on_dataset(golden_ds)
    
    print("\n--- STEP 2: Running RAGAS Evaluation Framework ---")
    try:
        result = evaluator.evaluate_results(raw_df, metric_names=metrics)
    except Exception as e:
        print(f"\nEvaluation failed with error: {e}")
        sys.exit(1)
        
    print("\n--- STEP 3: Generating Evaluation Report ---")
    reports_dir = settings.paths.golden_dataset.parent / "eval_reports"
    report_path = generate_markdown_report(result, raw_df, reports_dir)
    print(f"Saved evaluation report to: {report_path.resolve()}")

    summary_scores = extract_summary_scores(result)
    if not summary_scores:
        print("\nError: Could not extract numeric metric scores from evaluation result.")
        sys.exit(1)
    
    print("\n--- STEP 4: Enforcing Quality Thresholds ---")
    print("Average Evaluation Scores:")
    for metric, score in summary_scores.items():
        print(f" - {metric}: {score:.4f}")
        
    # Check thresholds
    failed = False
    
    # Faithfulness check
    faith_score = summary_scores.get("faithfulness", 0.0)
    faith_threshold = settings.eval.faithfulness_threshold
    if faith_score < faith_threshold:
        print(f"❌ FAIL: Faithfulness score ({faith_score:.4f}) is below threshold ({faith_threshold:.2f})")
        failed = True
    else:
        print(f"✅ PASS: Faithfulness score ({faith_score:.4f}) meets threshold ({faith_threshold:.2f})")
        
    # Answer Relevancy check
    rel_score = summary_scores.get("answer_relevancy", 0.0)
    rel_threshold = settings.eval.answer_relevancy_threshold
    if rel_score < rel_threshold:
        print(f"❌ FAIL: Answer Relevancy score ({rel_score:.4f}) is below threshold ({rel_threshold:.2f})")
        failed = True
    else:
        print(f"✅ PASS: Answer Relevancy score ({rel_score:.4f}) meets threshold ({rel_threshold:.2f})")
        
    if failed:
        print("\n❌ Build/Test Gate: FAILED (Regressions detected).")
        sys.exit(1)
    else:
        print("\n✅ Build/Test Gate: PASSED.")
        sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGGED - RAG Quality Evaluation Runner"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N samples (useful for quick smoke tests with local LLMs)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="groq",
        help="LLM provider for generation + RAGAS eval (default: groq)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-8b-instant",
        help="Model name for the chosen provider (default: llama-3.1-8b-instant)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        choices=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        metavar="METRIC",
        help="Metrics to compute (default: all). For fast iteration use the gated "
             "pair: --metrics faithfulness answer_relevancy",
    )
    args = parser.parse_args()
    run_evaluation(
        num_samples=args.num_samples,
        provider=args.provider,
        model=args.model,
        metrics=args.metrics,
    )


if __name__ == "__main__":
    main()
