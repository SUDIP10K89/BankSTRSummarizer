
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_PATH = PROJECT_ROOT / "data" / "track6_reports_dataset.csv"
MINIMAL_NARRATIVE = "Suspicious transaction observed."


def text_or_none(element: ET.Element | None, path: str) -> str | None:
    if element is None:
        return None

    value = element.findtext(path)
    if value is None:
        return None

    value = value.strip()
    return value or None


def classify_report_type(narrative: str | None) -> str:
    cleaned = (narrative or "").strip()
    if not cleaned or cleaned == MINIMAL_NARRATIVE:
        return "Type-A"
    return "Type-B"


def parse_report(xml_file: Path) -> dict[str, Any]:
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    transaction = root.find("transaction")

    narrative = text_or_none(root, "reason")
    amount_local = text_or_none(transaction, "amount_local")

    row: dict[str, Any] = {
        "source_file": xml_file.name,
        "report_id": text_or_none(root, "report_id"),
        "entity_reference": text_or_none(root, "entity_reference"),
        "submission_date": text_or_none(root, "submission_date"),
        "currency_code": text_or_none(root, "currency_code_local"),
        "narrative": narrative,
        "narrative_length": len(narrative or ""),
        "report_type": classify_report_type(narrative),
        "transaction_number": text_or_none(transaction, "transactionnumber"),
        "amount_local": float(amount_local) if amount_local is not None else None,
        "date_transaction": text_or_none(transaction, "date_transaction"),
        "transaction_location": text_or_none(transaction, "transaction_location"),
        "transaction_mode_code": text_or_none(transaction, "transmode_code"),
        "transaction_mode": text_or_none(transaction, "transmode_comment"),
        "from_institution": text_or_none(
            transaction, ".//t_from_my_client/from_account/institution_name"
        ),
        "from_account_name": text_or_none(
            transaction, ".//t_from_my_client/from_account/account_name"
        ),
        "from_account_number": text_or_none(
            transaction, ".//t_from_my_client/from_account/account"
        ),
        "from_country": text_or_none(transaction, ".//t_from_my_client/from_country"),
        "to_institution": text_or_none(transaction, ".//t_to/to_account/institution_name"),
        "to_account_name": text_or_none(transaction, ".//t_to/to_account/account_name"),
        "to_account_number": text_or_none(transaction, ".//t_to/to_account/account"),
        "to_country": text_or_none(transaction, ".//t_to/to_country"),
    }

    return row


def load_reports(reports_dir: Path = REPORTS_DIR) -> pd.DataFrame:
    xml_files = sorted(reports_dir.glob("report_*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No report_*.xml files found in {reports_dir}")

    rows = [parse_report(xml_file) for xml_file in xml_files]
    return pd.DataFrame(rows)


def save_dataset(df: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    print("Track 6 data pipeline complete")
    print(f"Reports parsed: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Report types:")
    for report_type, count in df["report_type"].value_counts().sort_index().items():
        print(f"  {report_type}: {count}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Track 6 STR dataset.")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory containing report_*.xml files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="CSV path for the parsed dataset.",
    )
    args = parser.parse_args()

    df = load_reports(args.reports_dir)
    save_dataset(df, args.output)
    print_summary(df, args.output)


if __name__ == "__main__":
    main()
