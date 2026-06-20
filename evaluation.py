"""
Task 1.3 evaluation framework for Track 6 STR summarization.

Evaluates generated summaries for:
- 100-200 word length compliance
- entity/fact preservation
- optional ROUGE-L and BERTScore when gold summaries are available
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_PATH = PROJECT_ROOT / "data" / "track6_reports_with_entities.csv"
SUMMARIES_PATH = PROJECT_ROOT / "data" / "track6_llm_summaries.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "track6_evaluation_results.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "track6_evaluation_summary.json"
DEFAULT_SUMMARY_COLUMN = "summary"
EVALUATION_COLUMNS = [
    "word_count",
    "length_ok",
    "has_customer",
    "has_counterparty",
    "has_banks",
    "has_amount",
    "has_date",
    "has_transaction_mode",
    "has_account_numbers",
    "critical_entity_preservation_ok",
    "strict_entity_preservation_ok",
    "entity_preservation_ok",
    "overall_ok",
    "rouge_l_f1",
    "bertscore_f1",
]


def clean(value: Any, fallback: str = "") -> str:
    """Return normalized text for matching."""
    if value is None or pd.isna(value):
        return fallback

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback

    return re.sub(r"\s+", " ", text)


def word_count(text: Any) -> int:
    """Count simple word tokens."""
    return len(re.findall(r"\b\w+\b", clean(text)))


def split_entities(value: Any) -> list[str]:
    """Split semicolon-delimited entity columns."""
    text = clean(value)
    if not text:
        return []

    return [item.strip() for item in text.split(";") if item.strip()]


def load_entities_json(value: Any) -> dict[str, Any]:
    """Parse the entities_json column when available."""
    text = clean(value)
    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def normalize_for_contains(text: Any) -> str:
    """Normalize text for case-insensitive containment checks."""
    return clean(text).casefold()


def contains_all(summary: Any, expected_values: Iterable[Any]) -> bool:
    """Return True when all expected string values appear in the summary."""
    summary_text = normalize_for_contains(summary)
    values = [normalize_for_contains(value) for value in expected_values]
    values = [value for value in values if value]
    if not values:
        return True

    return all(value in summary_text for value in values)


def extract_numeric_amounts(text: Any) -> list[Decimal]:
    """Extract amount-like numbers, accepting comma or plain formatting."""
    cleaned = clean(text)
    if not cleaned:
        return []

    amounts: list[Decimal] = []
    for match in re.findall(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", cleaned):
        try:
            amounts.append(Decimal(match.replace(",", "")))
        except InvalidOperation:
            continue

    return amounts


def contains_amounts(summary: Any, expected_amounts: Iterable[Any]) -> bool:
    """Compare amount preservation by numeric value, not display formatting."""
    expected_numbers: list[Decimal] = []
    for amount in expected_amounts:
        expected_numbers.extend(extract_numeric_amounts(amount))

    if not expected_numbers:
        return True

    summary_numbers = extract_numeric_amounts(summary)
    return all(
        any(abs(found - expected) <= Decimal("0.01") for found in summary_numbers)
        for expected in expected_numbers
    )


def tokenize_for_rouge(text: Any) -> list[str]:
    """Tokenize text for ROUGE-L."""
    return re.findall(r"\w+", clean(text).casefold())


def lcs_length(left: list[str], right: list[str]) -> int:
    """Compute longest common subsequence length using a compact DP table."""
    if not left or not right:
        return 0

    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current

    return previous[-1]


def rouge_l_f1(candidate: Any, reference: Any) -> float | None:
    """Compute ROUGE-L F1 against a gold reference summary."""
    candidate_tokens = tokenize_for_rouge(candidate)
    reference_tokens = tokenize_for_rouge(reference)
    if not candidate_tokens or not reference_tokens:
        return None

    lcs = lcs_length(candidate_tokens, reference_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def evaluate_row(row: pd.Series, summary_column: str) -> dict[str, Any]:
    """Evaluate one generated summary."""
    summary = row.get(summary_column)
    entities = load_entities_json(row.get("entities_json"))

    customer_names = split_entities(row.get("entities_customer_names"))
    counterparty_names = split_entities(row.get("entities_counterparty_names"))
    bank_names = split_entities(row.get("entities_bank_names"))
    amounts = split_entities(row.get("entities_amounts"))
    dates = split_entities(row.get("entities_dates"))
    transaction_modes = split_entities(row.get("entities_transaction_modes"))
    account_numbers = entities.get("account_numbers", [])

    words = word_count(summary)
    checks = {
        "word_count": words,
        "length_ok": 100 <= words <= 200,
        "has_customer": contains_all(summary, customer_names),
        "has_counterparty": contains_all(summary, counterparty_names),
        "has_banks": contains_all(summary, bank_names),
        "has_amount": contains_amounts(summary, amounts),
        "has_date": contains_all(summary, dates),
        "has_transaction_mode": contains_all(summary, transaction_modes),
        "has_account_numbers": contains_all(summary, account_numbers),
    }
    critical_keys = [
        "has_customer",
        "has_counterparty",
        "has_banks",
        "has_amount",
        "has_date",
        "has_transaction_mode",
    ]
    strict_keys = critical_keys + ["has_account_numbers"]

    checks["critical_entity_preservation_ok"] = all(checks[key] for key in critical_keys)
    checks["strict_entity_preservation_ok"] = all(checks[key] for key in strict_keys)
    checks["overall_ok"] = checks["length_ok"] and checks["critical_entity_preservation_ok"]

    return checks


def add_reference_metrics(
    results: pd.DataFrame,
    summary_column: str,
    reference_column: str | None,
) -> pd.DataFrame:
    """Add ROUGE-L and optional BERTScore if a gold reference column exists."""
    if not reference_column or reference_column not in results.columns:
        return results

    evaluated = results.copy()
    evaluated["rouge_l_f1"] = evaluated.apply(
        lambda row: rouge_l_f1(row.get(summary_column), row.get(reference_column)),
        axis=1,
    )

    try:
        from bert_score import score as bert_score
    except ImportError:
        evaluated["bertscore_f1"] = None
        return evaluated

    valid = evaluated[summary_column].notna() & evaluated[reference_column].notna()
    if not valid.any():
        evaluated["bertscore_f1"] = None
        return evaluated

    _, _, f1_scores = bert_score(
        evaluated.loc[valid, summary_column].astype(str).tolist(),
        evaluated.loc[valid, reference_column].astype(str).tolist(),
        lang="en",
        verbose=False,
    )
    evaluated["bertscore_f1"] = None
    evaluated.loc[valid, "bertscore_f1"] = [float(value) for value in f1_scores]
    return evaluated


def evaluate_summaries(
    reports: pd.DataFrame,
    summaries: pd.DataFrame,
    summary_column: str,
    reference_column: str | None = None,
) -> pd.DataFrame:
    """Join report facts to generated summaries and evaluate each row."""
    if summary_column not in summaries.columns:
        raise ValueError(f"Summary column not found: {summary_column}")

    summaries = summaries.drop(
        columns=[column for column in EVALUATION_COLUMNS if column in summaries.columns]
    )
    merged = summaries.merge(reports, on="report_id", how="left", suffixes=("", "_report"))
    check_rows = [evaluate_row(row, summary_column) for _, row in merged.iterrows()]
    evaluated = pd.concat([merged, pd.DataFrame(check_rows)], axis=1)
    return add_reference_metrics(evaluated, summary_column, reference_column)


def summarize_results(results: pd.DataFrame) -> dict[str, Any]:
    """Create aggregate metrics for reporting."""
    total = len(results)
    if total == 0:
        return {"total_summaries": 0}

    boolean_columns = [
        "length_ok",
        "has_customer",
        "has_counterparty",
        "has_banks",
        "has_amount",
        "has_date",
        "has_transaction_mode",
        "has_account_numbers",
        "critical_entity_preservation_ok",
        "strict_entity_preservation_ok",
        "overall_ok",
    ]

    summary: dict[str, Any] = {
        "total_summaries": total,
        "average_word_count": round(float(results["word_count"].mean()), 2),
        "min_word_count": int(results["word_count"].min()),
        "max_word_count": int(results["word_count"].max()),
    }

    for column in boolean_columns:
        summary[f"{column}_rate"] = round(float(results[column].mean()), 4)

    if "rouge_l_f1" in results.columns and results["rouge_l_f1"].notna().any():
        summary["average_rouge_l_f1"] = round(float(results["rouge_l_f1"].mean()), 4)

    if "bertscore_f1" in results.columns and results["bertscore_f1"].notna().any():
        summary["average_bertscore_f1"] = round(float(results["bertscore_f1"].mean()), 4)

    return summary


def print_summary(summary: dict[str, Any], output_path: Path, summary_path: Path) -> None:
    """Print compact evaluation summary."""
    print("Track 6 evaluation complete")
    print(f"Summaries evaluated: {summary.get('total_summaries', 0)}")
    print(f"Average word count: {summary.get('average_word_count', 'N/A')}")
    print(f"Length pass rate: {summary.get('length_ok_rate', 'N/A')}")
    print(
        "Critical entity preservation pass rate: "
        f"{summary.get('critical_entity_preservation_ok_rate', 'N/A')}"
    )
    print(f"Overall pass rate: {summary.get('overall_ok_rate', 'N/A')}")
    print(f"Results: {output_path}")
    print(f"Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Track 6 summaries.")
    parser.add_argument("--reports", type=Path, default=REPORTS_PATH)
    parser.add_argument("--summaries", type=Path, default=SUMMARIES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--summary-column", default=DEFAULT_SUMMARY_COLUMN)
    parser.add_argument(
        "--reference-column",
        default=None,
        help="Optional gold summary column for ROUGE-L/BERTScore.",
    )
    args = parser.parse_args()

    reports = pd.read_csv(args.reports)
    summaries = pd.read_csv(args.summaries)
    results = evaluate_summaries(
        reports,
        summaries,
        summary_column=args.summary_column,
        reference_column=args.reference_column,
    )
    aggregate = summarize_results(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    args.summary_output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print_summary(aggregate, args.output, args.summary_output)


if __name__ == "__main__":
    main()
