"""
Build demo-ready Track 6 output.

Creates a compact sample table with original narrative, extracted facts,
generated summary, validation result, and pass/fail explanation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATION_PATH = PROJECT_ROOT / "data" / "track6_evaluation_results.csv"
DEMO_CSV_PATH = PROJECT_ROOT / "data" / "track6_demo_sample.csv"
DEMO_MD_PATH = PROJECT_ROOT / "data" / "track6_demo_sample.md"


FACT_COLUMNS = [
    "entities_customer_names",
    "entities_counterparty_names",
    "entities_bank_names",
    "entities_amounts",
    "entities_dates",
    "entities_transaction_modes",
    "from_account_number",
    "to_account_number",
]


CHECK_COLUMNS = [
    "length_ok",
    "has_customer",
    "has_counterparty",
    "has_banks",
    "has_amount",
    "has_date",
    "has_transaction_mode",
    "has_account_numbers",
]


def clean(value: Any, fallback: str = "Not available") -> str:
    """Return a display-safe string."""
    if value is None or pd.isna(value):
        return fallback

    text = str(value).strip()
    return text if text else fallback


def truncate(text: Any, max_chars: int = 500) -> str:
    """Shorten long narratives for demo readability."""
    cleaned = clean(text)
    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[: max_chars - 3].rstrip() + "..."


def format_facts(row: pd.Series) -> str:
    """Format extracted facts as a compact presentation string."""
    facts = {
        "Customer": row.get("entities_customer_names"),
        "Counterparty": row.get("entities_counterparty_names"),
        "Banks": row.get("entities_bank_names"),
        "Amount": row.get("entities_amounts"),
        "Date": row.get("entities_dates"),
        "Mode": row.get("entities_transaction_modes"),
        "From account": row.get("from_account_number"),
        "To account": row.get("to_account_number"),
    }
    return " | ".join(f"{label}: {clean(value)}" for label, value in facts.items())


def explain_validation(row: pd.Series) -> str:
    """Create a plain-English pass/fail explanation."""
    failures = [column for column in CHECK_COLUMNS if not bool(row.get(column))]
    if bool(row.get("overall_ok")) and not failures:
        return (
            f"PASS: {int(row['word_count'])} words, within 100-200; "
            "all critical facts and account numbers preserved."
        )

    messages: list[str] = []
    if not bool(row.get("length_ok")):
        messages.append(f"word count is {int(row['word_count'])}, outside 100-200")

    missing_map = {
        "has_customer": "customer",
        "has_counterparty": "counterparty",
        "has_banks": "bank/institution",
        "has_amount": "amount",
        "has_date": "date",
        "has_transaction_mode": "transaction mode",
        "has_account_numbers": "account numbers",
    }
    missing = [missing_map[column] for column in failures if column in missing_map]
    if missing:
        messages.append("missing " + ", ".join(missing))

    return "FAIL: " + "; ".join(messages)


def build_demo_table(evaluation: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Select and format demo rows."""
    sampled = evaluation.head(sample_size).copy()
    demo = pd.DataFrame(
        {
            "report_id": sampled["report_id"],
            "report_type": sampled["report_type"],
            "original_narrative": sampled["narrative"].map(lambda value: truncate(value, 700)),
            "extracted_facts": sampled.apply(format_facts, axis=1),
            "generated_summary": sampled["llm_summary"],
            "validation_result": sampled["overall_ok"].map(
                lambda passed: "PASS" if bool(passed) else "FAIL"
            ),
            "pass_fail_explanation": sampled.apply(explain_validation, axis=1),
        }
    )
    return demo


def write_markdown(demo: pd.DataFrame, output_path: Path) -> None:
    """Write a readable Markdown demo artifact."""
    lines = [
        "# Track 6 Demo Sample",
        "",
        "Demo-ready STR summarization examples with extracted facts and validation results.",
        "",
    ]

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
    parser = argparse.ArgumentParser(description="Build Track 6 demo sample output.")
    parser.add_argument("--evaluation", type=Path, default=EVALUATION_PATH)
    parser.add_argument("--output-csv", type=Path, default=DEMO_CSV_PATH)
    parser.add_argument("--output-md", type=Path, default=DEMO_MD_PATH)
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()

    evaluation = pd.read_csv(args.evaluation)
    demo = build_demo_table(evaluation, sample_size=args.sample_size)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(args.output_csv, index=False)
    write_markdown(demo, args.output_md)

    print("Track 6 demo output complete")
    print(f"Rows: {len(demo)}")
    print(f"CSV: {args.output_csv}")
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
