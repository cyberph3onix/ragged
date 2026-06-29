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
from evaluation.report import generate_markdown_report


def run_evaluation() -> None:
    dataset_path = settings.paths.golden_dataset
    print(f"Loading benchmark dataset from: {dataset_path}")
    
    try:
        golden_ds = GoldenDataset.load_from_json(dataset_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
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
        result = evaluator.evaluate_results(raw_df)
    except Exception as e:
        print(f"\nEvaluation failed with error: {e}")
        sys.exit(1)
        
    print("\n--- STEP 3: Generating Evaluation Report ---")
    reports_dir = settings.paths.golden_dataset.parent / "eval_reports"
    report_path = generate_markdown_report(result, raw_df, reports_dir)
    print(f"Saved evaluation report to: {report_path.resolve()}")
    
    print("\n--- STEP 4: Enforcing Quality Thresholds ---")
    print("Average Evaluation Scores:")
    for metric, score in result.items():
        print(f" - {metric}: {score:.4f}")
        
    # Check thresholds
    failed = False
    
    # Faithfulness check
    faith_score = result.get("faithfulness", 0.0)
    faith_threshold = settings.eval.faithfulness_threshold
    if faith_score < faith_threshold:
        print(f"❌ FAIL: Faithfulness score ({faith_score:.4f}) is below threshold ({faith_threshold:.2f})")
        failed = True
    else:
        print(f"✅ PASS: Faithfulness score ({faith_score:.4f}) meets threshold ({faith_threshold:.2f})")
        
    # Answer Relevancy check
    rel_score = result.get("answer_relevancy", 0.0)
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
    # Allows overriding of defaults via arguments if needed in the future
    parser.parse_args()
    run_evaluation()


if __name__ == "__main__":
    main()
