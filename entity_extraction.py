"""
Task 1.2 entity extraction for Track 6 STR summarization.

Uses structured fields from the parsed XML dataset as the source of truth, then
optionally enriches entities with spaCy findings from the free-text narrative.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "data" / "track6_reports_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "track6_reports_with_entities.csv"
DEFAULT_SPACY_MODEL = "en_core_web_sm"


def clean_value(value: Any) -> str | None:
    """Normalize a scalar value from pandas/spaCy into a clean string."""
    if value is None or pd.isna(value):
        return None

    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() == "nan":
        return None

    return re.sub(r"\s+", " ", cleaned)


def unique(values: Iterable[Any]) -> list[str]:
    """Return non-empty unique strings while preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        cleaned = clean_value(value)
        if cleaned is None:
            continue

        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


def date_only(value: Any) -> str | None:
    """Reduce ISO datetimes to YYYY-MM-DD dates for easier matching."""
    cleaned = clean_value(value)
    if cleaned is None:
        return None

    return cleaned.split("T", 1)[0]


def amount_with_currency(amount: Any, currency: Any) -> str | None:
    """Format amount and currency as a summary-ready entity string."""
    amount_text = clean_value(amount)
    if amount_text is None:
        return None

    currency_text = clean_value(currency) or "NPR"
    return f"{currency_text} {amount_text}"


def load_spacy_model(model_name: str):
    """Load spaCy model when installed; return None when unavailable."""
    try:
        import spacy
    except ImportError:
        print("spaCy is not installed; using structured-field extraction only.")
        return None

    try:
        return spacy.load(model_name)
    except OSError:
        print(f"spaCy model '{model_name}' is not installed; using structured fields only.")
        return None


def extract_spacy_entities(narrative: Any, nlp) -> dict[str, list[str]]:
    """Extract candidate entities from narrative text using spaCy."""
    if nlp is None:
        return {
            "spacy_persons": [],
            "spacy_orgs": [],
            "spacy_money": [],
            "spacy_dates": [],
            "spacy_locations": [],
        }

    text = clean_value(narrative)
    if text is None:
        return {
            "spacy_persons": [],
            "spacy_orgs": [],
            "spacy_money": [],
            "spacy_dates": [],
            "spacy_locations": [],
        }

    doc = nlp(text)
    return {
        "spacy_persons": unique(ent.text for ent in doc.ents if ent.label_ == "PERSON"),
        "spacy_orgs": unique(ent.text for ent in doc.ents if ent.label_ == "ORG"),
        "spacy_money": unique(ent.text for ent in doc.ents if ent.label_ == "MONEY"),
        "spacy_dates": unique(ent.text for ent in doc.ents if ent.label_ == "DATE"),
        "spacy_locations": unique(ent.text for ent in doc.ents if ent.label_ == "GPE"),
    }


def extract_structured_entities(row: pd.Series) -> dict[str, list[str]]:
    """Extract high-confidence entities from parsed XML/DataFrame fields."""
    return {
        "customer_names": unique([row.get("from_account_name")]),
        "counterparty_names": unique([row.get("to_account_name")]),
        "bank_names": unique([row.get("from_institution"), row.get("to_institution")]),
        "amounts": unique(
            [amount_with_currency(row.get("amount_local"), row.get("currency_code"))]
        ),
        "dates": unique([date_only(row.get("date_transaction"))]),
        "transaction_modes": unique([row.get("transaction_mode")]),
        "countries": unique([row.get("from_country"), row.get("to_country")]),
        "account_numbers": unique(
            [row.get("from_account_number"), row.get("to_account_number")]
        ),
    }


def extract_entities(row: pd.Series, nlp=None) -> dict[str, list[str]]:
    """Create one entity bundle for a report row."""
    structured = extract_structured_entities(row)
    spacy_entities = extract_spacy_entities(row.get("narrative"), nlp)

    return {
        **structured,
        "candidate_names_from_narrative": unique(
            spacy_entities["spacy_persons"]
            + structured["customer_names"]
            + structured["counterparty_names"]
        ),
        "candidate_orgs_from_narrative": unique(
            spacy_entities["spacy_orgs"] + structured["bank_names"]
        ),
        "candidate_amounts_from_narrative": unique(
            structured["amounts"] + spacy_entities["spacy_money"]
        ),
        "candidate_dates_from_narrative": unique(
            structured["dates"] + spacy_entities["spacy_dates"]
        ),
        "candidate_locations_from_narrative": unique(
            structured["countries"] + spacy_entities["spacy_locations"]
        ),
    }


def add_entity_columns(df: pd.DataFrame, nlp=None) -> pd.DataFrame:
    """Add JSON entity bundles and convenient flat columns to the dataset."""
    enriched = df.copy()
    entity_bundles = [extract_entities(row, nlp) for _, row in enriched.iterrows()]

    enriched["entities_json"] = [
        json.dumps(bundle, ensure_ascii=False, sort_keys=True) for bundle in entity_bundles
    ]

    for field in [
        "customer_names",
        "counterparty_names",
        "bank_names",
        "amounts",
        "dates",
        "transaction_modes",
        "countries",
    ]:
        enriched[f"entities_{field}"] = [
            "; ".join(bundle[field]) for bundle in entity_bundles
        ]

    return enriched


def print_summary(df: pd.DataFrame, output_path: Path, used_spacy: bool) -> None:
    """Print a compact verification summary."""
    print("Track 6 entity extraction complete")
    print(f"Reports enriched: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"spaCy enrichment: {'enabled' if used_spacy else 'not available'}")
    print(
        "Rows with customer/counterparty entities: "
        f"{df['entities_customer_names'].ne('').sum()}/"
        f"{df['entities_counterparty_names'].ne('').sum()}"
    )
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Track 6 STR entities.")
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="CSV generated by data_pipeline.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="CSV path for the entity-enriched dataset.",
    )
    parser.add_argument(
        "--spacy-model",
        default=DEFAULT_SPACY_MODEL,
        help="spaCy model name for narrative entity extraction.",
    )
    parser.add_argument(
        "--no-spacy",
        action="store_true",
        help="Skip spaCy and use structured fields only.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    nlp = None if args.no_spacy else load_spacy_model(args.spacy_model)
    enriched = add_entity_columns(df, nlp)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    print_summary(enriched, args.output, used_spacy=nlp is not None)


if __name__ == "__main__":
    main()
