from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

REPORTS_PATH = PROJECT_ROOT / "data" / "track6_reports_with_entities.csv"
EXTRACTIVE_PATH = PROJECT_ROOT / "data" / "summary_extractive.csv"
HYBRID_PATH = PROJECT_ROOT / "data" / "summary_hybrid.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "track6_model_comparison_results.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "track6_model_comparison_summary.json"


# =========================
# LOAD DATA
# =========================
reports = pd.read_csv(REPORTS_PATH)

extractive = pd.read_csv(EXTRACTIVE_PATH)
hybrid = pd.read_csv(HYBRID_PATH)

extractive["model"] = "extractive"
hybrid["model"] = "hybrid"

summaries = pd.concat([extractive, hybrid], ignore_index=True)

df = summaries.merge(reports, on="report_id", how="left")


# =========================
# HELPERS (YOUR ORIGINAL LOGIC - CLEANED)
# =========================

def clean(value: Any, fallback: str = "") -> str:
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def word_count(text: Any) -> int:
    return len(re.findall(r"\b\w+\b", clean(text)))


def split_entities(value: Any) -> list[str]:
    text = clean(value)
    return [x.strip() for x in text.split(";") if x.strip()]


def normalize(text: Any) -> str:
    return clean(text).casefold()


def contains_all(summary: Any, expected: Iterable[Any]) -> bool:
    summary_text = normalize(summary)

    values = [normalize(v) for v in expected if v]
    if not values:
        return True

    return all(v in summary_text for v in values)


def extract_numbers(text: Any) -> list[Decimal]:
    cleaned = clean(text)
    nums = []

    for m in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", cleaned):
        try:
            nums.append(Decimal(m.replace(",", "")))
        except InvalidOperation:
            pass

    return nums


def contains_amounts(summary: Any, expected: Iterable[Any]) -> bool:
    summary_nums = extract_numbers(summary)

    expected_nums = []
    for e in expected:
        expected_nums.extend(extract_numbers(e))

    if not expected_nums:
        return True

    return all(
        any(abs(s - e) <= Decimal("0.01") for s in summary_nums)
        for e in expected_nums
    )


# =========================
# EVALUATION FUNCTION
# =========================

def evaluate_row(row: pd.Series) -> dict[str, Any]:
    summary = row["summary"]

    customers = split_entities(row.get("entities_customer_names"))
    counterparties = split_entities(row.get("entities_counterparty_names"))
    banks = split_entities(row.get("entities_bank_names"))
    amounts = split_entities(row.get("entities_amounts"))
    dates = split_entities(row.get("entities_dates"))
    modes = split_entities(row.get("entities_transaction_modes"))

    wc = word_count(summary)

    result = {
        "report_id": row["report_id"],
        "model": row["model"],

        "word_count": wc,
        "length_ok": 100 <= wc <= 200,

        "has_customer": contains_all(summary, customers),
        "has_counterparty": contains_all(summary, counterparties),
        "has_bank": contains_all(summary, banks),
        "has_amount": contains_amounts(summary, amounts),
        "has_date": contains_all(summary, dates),
        "has_mode": contains_all(summary, modes),
    }

    critical = [
        "has_customer",
        "has_counterparty",
        "has_bank",
        "has_amount",
        "has_date",
        "has_mode",
    ]

    result["critical_ok"] = all(result[c] for c in critical)
    result["overall_ok"] = result["length_ok"] and result["critical_ok"]

    return result


# =========================
# RUN EVALUATION
# =========================
results = pd.DataFrame([evaluate_row(row) for _, row in df.iterrows()])


# =========================
# MODEL COMPARISON
# =========================
comparison = results.groupby("model").mean(numeric_only=True)

print("\n===== MODEL COMPARISON =====\n")
print(comparison)


# =========================
# SAVE RESULTS
# =========================
results.to_csv(OUTPUT_PATH, index=False)
comparison.to_json(SUMMARY_PATH, indent=2)

print("\nSaved results to:", OUTPUT_PATH)
print("Saved summary to:", SUMMARY_PATH)