"""
Task 3.2 LLM-based STR summarization with LangChain + Hugging Face.

Uses meta-llama/Llama-3.1-70B-Instruct through Hugging Face inference, grounded
by the entity-enriched dataset from Task 1.2.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "data" / "track6_reports_with_entities.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "track6_llm_summaries.csv"
MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"


SYSTEM_PROMPT = """You are a financial crime analyst writing neutral STR summaries.
Preserve facts exactly. Do not invent names, amounts, dates, banks, locations, or legal conclusions."""


USER_PROMPT_TEMPLATE = """Write a concise Suspicious Transaction Report summary in 100-200 words.

Rules:
- Include customer name, counterparty, bank/institution names, amount, date, and transaction mode when available.
- Use only the structured facts and narrative below.
- If the narrative is minimal, summarize using the structured facts only.
- Do not say the customer committed a crime.
- Keep a neutral analyst-facing tone.

Structured facts:
Report ID: {report_id}
Report type: {report_type}
Customer: {customer}
Counterparty: {counterparty}
Banks: {banks}
Amount: {amount}
Date: {date}
Transaction mode: {transaction_mode}
Countries: {countries}

Narrative:
{narrative}

Return only the summary."""


def clean(value: Any, fallback: str = "Not available") -> str:
    """Return clean display text for prompt fields."""
    if value is None or pd.isna(value):
        return fallback

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback

    return re.sub(r"\s+", " ", text)


def build_user_prompt(row: pd.Series) -> str:
    """Create the report-specific prompt body."""
    return USER_PROMPT_TEMPLATE.format(
        report_id=clean(row.get("report_id")),
        report_type=clean(row.get("report_type")),
        customer=clean(row.get("entities_customer_names")),
        counterparty=clean(row.get("entities_counterparty_names")),
        banks=clean(row.get("entities_bank_names")),
        amount=clean(row.get("entities_amounts")),
        date=clean(row.get("entities_dates")),
        transaction_mode=clean(row.get("entities_transaction_modes")),
        countries=clean(row.get("entities_countries")),
        narrative=clean(row.get("narrative")),
    )


def build_llama_prompt(row: pd.Series) -> str:
    """Format the prompt with Llama 3.x instruct chat tokens for dry runs."""
    user_prompt = build_user_prompt(row)
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_messages(row: pd.Series) -> list[tuple[str, str]]:
    """Create LangChain chat messages for the Hugging Face chat wrapper."""
    return [
        ("system", SYSTEM_PROMPT),
        ("human", build_user_prompt(row)),
    ]


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load simple KEY=VALUE entries from .env without adding a dependency."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_llm(model_id: str, max_new_tokens: int, temperature: float, top_p: float):
    """Create a LangChain Hugging Face endpoint client."""
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install langchain-huggingface to run LLM summaries."
        ) from exc

    load_dotenv()
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN."
        )

    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task="conversational",
        huggingfacehub_api_token=token,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        return_full_text=False,
    )
    return ChatHuggingFace(
        llm=endpoint,
        model_id=model_id,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def word_count(text: str) -> int:
    """Count simple word tokens for the 100-200 word constraint."""
    return len(re.findall(r"\b\w+\b", text or ""))


def contains_value(summary: str, value: Any) -> bool:
    """Case-insensitive containment check for required structured facts."""
    value_text = clean(value, fallback="")
    if not value_text:
        return True

    return value_text.casefold() in summary.casefold()


def validate_summary(summary: str, row: pd.Series) -> dict[str, Any]:
    """Lightweight factual-preservation checks for generated output."""
    words = word_count(summary)
    checks = {
        "word_count": words,
        "length_ok": 100 <= words <= 200,
        "has_customer": contains_value(summary, row.get("entities_customer_names")),
        "has_counterparty": contains_value(summary, row.get("entities_counterparty_names")),
        "has_amount": contains_value(summary, row.get("entities_amounts")),
        "has_date": contains_value(summary, row.get("entities_dates")),
        "has_transaction_mode": contains_value(
            summary, row.get("entities_transaction_modes")
        ),
    }
    checks["entity_preservation_ok"] = all(
        checks[key]
        for key in [
            "has_customer",
            "has_counterparty",
            "has_amount",
            "has_date",
            "has_transaction_mode",
        ]
    )
    return checks


def select_rows(df: pd.DataFrame, limit: int | None, all_rows: bool) -> pd.DataFrame:
    """Pick rows to summarize, protecting against accidental full 70B runs."""
    if all_rows:
        return df

    if limit is None:
        limit = 3

    return df.head(limit)


def generate_summaries(
    df: pd.DataFrame,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> pd.DataFrame:
    """Generate and validate summaries for selected rows."""
    llm = load_llm(model_id, max_new_tokens, temperature, top_p)
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        response = llm.invoke(build_messages(row))
        summary = clean(response.content, fallback="")
        checks = validate_summary(summary, row)

        rows.append(
            {
                "report_id": row.get("report_id"),
                "report_type": row.get("report_type"),
                "model_id": model_id,
                "llm_summary": summary,
                **checks,
            }
        )

        print(
            f"Generated {row.get('report_id')}: "
            f"{checks['word_count']} words, "
            f"entities_ok={checks['entity_preservation_ok']}"
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Track 6 STR summaries with a Hugging Face LLM."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--all", action="store_true", help="Summarize all rows.")
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first prompt without calling the model.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    selected = select_rows(df, args.limit, args.all)

    if args.dry_run:
        print(build_llama_prompt(selected.iloc[0]))
        print(f"\nDry run only. Rows selected: {len(selected)}")
        return

    summaries = generate_summaries(
        selected,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(args.output, index=False)
    print(f"Saved summaries: {args.output}")


if __name__ == "__main__":
    main()
