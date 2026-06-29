"""
src/evaluation/report.py

Generates detailed evaluation reports in Markdown format.
"""

from datetime import datetime
from pathlib import Path
import pandas as pd
from config import settings


def generate_markdown_report(result, raw_df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Generates a markdown report summarizing evaluation results and saves it.

    Args:
        result: The Ragas Result object.
        raw_df: The DataFrame containing inputs (question, contexts, answer, ground_truth).
        output_dir: Folder to save reports into.

    Returns:
        Path to the saved report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"report_{timestamp}.md"

    # 1. Summary of overall average scores
    summary_rows = []
    failed_metrics = []
    
    # In Ragas, the result is dict-like
    for metric, score in result.items():
        threshold = None
        if metric == "faithfulness":
            threshold = settings.eval.faithfulness_threshold
        elif metric == "answer_relevancy":
            threshold = settings.eval.answer_relevancy_threshold

        status = "✅ PASS"
        if threshold is not None and score < threshold:
            status = "❌ FAIL"
            failed_metrics.append(metric)

        summary_rows.append(
            f"| **{metric}** | {score:.4f} | {f'{threshold:.2f}' if threshold else 'N/A'} | {status} |"
        )

    summary_table = "\n".join(summary_rows)

    # 2. Extract detailed scores per row
    try:
        scores_df = result.to_pandas()
    except Exception:
        scores_df = pd.DataFrame()

    # 3. Detailed breakdown of Q&A with scores
    qa_details = []
    for idx, row in raw_df.iterrows():
        # Get scores for this specific row if available
        scores_str = "N/A"
        if not scores_df.empty and idx < len(scores_df):
            row_scores = scores_df.iloc[idx]
            scores_parts = []
            for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if m in row_scores:
                    val = row_scores[m]
                    # Check for NaN / None
                    val_str = f"{val:.2f}" if pd.notna(val) else "N/A"
                    scores_parts.append(f"{m[:2].upper()}: {val_str}")
            scores_str = " | ".join(scores_parts)

        # Truncate context string for readability in report
        contexts_bullets = "\n".join(f"- Chunk {i+1}: \"{c[:150]}...\"" for i, c in enumerate(row["contexts"]))

        qa_details.append(
            f"### Case {idx+1}\n\n"
            f"**Question:** {row['question']}\n\n"
            f"**Scores:** `{scores_str}`\n\n"
            f"**Ground Truth:**\n> {row['ground_truth']}\n\n"
            f"**Generated Answer:**\n> {row['answer']}\n\n"
            f"**Retrieved Contexts:**\n{contexts_bullets}\n\n"
            f"---"
        )

    qa_section = "\n\n".join(qa_details)

    report_content = f"""# RAG Evaluation Report - {timestamp}

## System Settings Configured
- **Retrieval Mode:** `{settings.retrieval.mode}`
- **Reranker Enabled:** `{settings.reranker.enabled}` (Model: `{settings.reranker.model}`)
- **LLM Provider:** `{settings.llm.provider}` (Model: `{settings.llm.model}`)
- **Top-K Chunks:** `{settings.retrieval.top_k}`

## Executive Summary
| Metric | Average Score | Target Threshold | Status |
| --- | --- | --- | --- |
{summary_table}

{("### ⚠️ Regressions Detected\nThe following metrics did not meet target thresholds: " + ", ".join(f"`{m}`" for m in failed_metrics)) if failed_metrics else "### ✅ All metrics met or exceeded target thresholds."}

## Detailed Case Breakdown
{qa_section}
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_path
