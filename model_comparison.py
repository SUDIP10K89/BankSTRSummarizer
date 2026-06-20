"""
Compare Track 6 summarization approaches from evaluation summary files.

Inputs are the JSON outputs produced by evaluation.py for each model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EVALUATIONS = {
    "llm": PROJECT_ROOT / "data" / "track6_evaluation_summary.json",
    "extractive": PROJECT_ROOT / "data" / "track6_extractive_evaluation_summary.json",
    "hybrid": PROJECT_ROOT / "data" / "track6_hybrid_evaluation_summary.json",
}
OUTPUT_CSV = PROJECT_ROOT / "data" / "track6_model_comparison_results.csv"
OUTPUT_JSON = PROJECT_ROOT / "data" / "track6_model_comparison_summary.json"


COMPARISON_COLUMNS = [
    "model",
    "total_summaries",
    "average_word_count",
    "length_ok_rate",
    "critical_entity_preservation_ok_rate",
    "strict_entity_preservation_ok_rate",
    "overall_ok_rate",
    "has_customer_rate",
    "has_counterparty_rate",
    "has_banks_rate",
    "has_amount_rate",
    "has_date_rate",
    "has_transaction_mode_rate",
    "has_account_numbers_rate",
    "average_rouge_l_f1",
    "average_bertscore_f1",
]


def load_summary(model: str, path: Path) -> dict[str, Any]:
    """Load one model's aggregate evaluation summary."""
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation summary for {model}: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return {"model": model, **data}


def build_comparison(evaluations: dict[str, Path]) -> pd.DataFrame:
    """Build a sorted model-comparison table."""
    rows = [load_summary(model, path) for model, path in evaluations.items()]
    comparison = pd.DataFrame(rows)

    for column in COMPARISON_COLUMNS:
        if column not in comparison.columns:
            comparison[column] = None

    comparison = comparison[COMPARISON_COLUMNS]
    comparison = comparison.sort_values(
        by=["overall_ok_rate", "critical_entity_preservation_ok_rate", "length_ok_rate"],
        ascending=False,
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Track 6 model evaluations.")
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    args = parser.parse_args()

    comparison = build_comparison(DEFAULT_EVALUATIONS)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_csv, index=False)

    payload = {
        "best_model": comparison.iloc[0]["model"] if not comparison.empty else None,
        "models": comparison.to_dict(orient="records"),
    }
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Track 6 model comparison complete")
    print(comparison.to_string(index=False))
    print(f"CSV: {args.output_csv}")
    print(f"JSON: {args.output_json}")


if __name__ == "__main__":
    main()
