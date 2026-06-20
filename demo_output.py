"""
Build a demo-ready Track 6 sample table.

The output is intended for final presentation: narrative, extracted facts,
generated summary, validation result, and pass/fail explanation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATION_PATH = PROJECT_ROOT / "data" / "track6_evaluation_results.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "track6_demo_sample.csv"
OUTPUT_MD = PROJECT_ROOT / "data" / "track6_demo_sample.md"

CHECK_COLUMNS = {
    "length_ok": "summary length outside 100-200 words",
    "has_customer": "customer missing",
    "has_counterparty": "counterparty missing",
    "has_banks": "bank/institution missing",
    "has_amount": "amount missing or mismatched",
    "has_date": "date missing",
    "has_transaction_mode": "transaction mode missing",
    "has_account_numbers": "account number missing",
}


def clean(value: Any, fallback: str = "Not available") -> str:
    """Return display-safe text."""
    if value is None or pd.isna(value):
        return fallback

    text = str(value).strip()
    return text if text else fallback


def truncate(value: Any, max_chars: int = 700) -> str:
    """Keep long narratives readable for a demo table."""
    text = clean(value)
    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3].rstrip() + "..."


def format_facts(row: pd.Series) -> str:
    """Format extracted facts into one compact field."""
    facts = [
        ("Customer", row.get("entities_customer_names")),
        ("Counterparty", row.get("entities_counterparty_names")),
        ("Banks", row.get("entities_bank_names")),
        ("Amount", row.get("entities_amounts")),
        ("Date", row.get("entities_dates")),
        ("Mode", row.get("entities_transaction_modes")),
        ("From account", row.get("from_account_number")),
        ("To account", row.get("to_account_number")),
    ]
    return " | ".join(f"{label}: {clean(value)}" for label, value in facts)


def explain_result(row: pd.Series) -> str:
    """Produce a human-readable validation explanation."""
    failures = [
        message
        for column, message in CHECK_COLUMNS.items()
        if column in row and not bool(row[column])
    ]
    if not failures:
        return (
            f"PASS: {int(row['word_count'])} words; all required facts and account "
            "numbers are preserved."
        )

    return f"FAIL: {int(row['word_count'])} words; " + "; ".join(failures) + "."


def build_demo(evaluation: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Create the demo table."""
    sample = evaluation.head(sample_size).copy()
    return pd.DataFrame(
        {
            "report_id": sample["report_id"],
            "report_type": sample["report_type"],
            "original_narrative": sample["narrative"].map(truncate),
            "extracted_facts": sample.apply(format_facts, axis=1),
            "generated_summary": sample["summary"],
            "validation_result": sample["overall_ok"].map(
                lambda value: "PASS" if bool(value) else "FAIL"
            ),
            "pass_fail_explanation": sample.apply(explain_result, axis=1),
        }
    )


def write_markdown(demo: pd.DataFrame, output_path: Path) -> None:
    """Write presentation-friendly markdown."""
    lines = ["# Track 6 Demo Sample", ""]

    for index, row in demo.iterrows():
        lines.extend(
            [
                f"## {index + 1}. {row['report_id']} ({row['report_type']})",
                "",
                f"**Validation:** {row['validation_result']}",
                "",
                f"**Explanation:** {row['pass_fail_explanation']}",
                "",
                "**Extracted Facts**",
                "",
                row["extracted_facts"],
                "",
                "**Generated Summary**",
                "",
                row["generated_summary"],
                "",
                "**Original Narrative**",
                "",
                row["original_narrative"],
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Track 6 demo output.")
    parser.add_argument("--evaluation", type=Path, default=EVALUATION_PATH)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=OUTPUT_MD)
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()

    evaluation = pd.read_csv(args.evaluation)
    demo = build_demo(evaluation, args.sample_size)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(args.output_csv, index=False)
    write_markdown(demo, args.output_md)

    print("Track 6 demo output complete")
    print(f"Rows: {len(demo)}")
    print(f"CSV: {args.output_csv}")
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
