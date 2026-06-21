

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
- Include every available customer name, counterparty, bank/institution name, amount, date, transaction mode, and account number.
- If a key fact is not available, say it is not available. If you intentionally exclude a key fact, state the reason briefly.
- Use only the structured facts and narrative below. Do not add facts from outside this input.
- If the narrative is minimal, summarize using the structured facts only.
- Do not say the customer committed a crime.
- Keep a neutral analyst-facing tone that can be read in under 30 seconds.
- The output must be one paragraph of 100-200 words.

Structured facts:
Report ID: {report_id}
Report type: {report_type}
Customer: {customer}
Counterparty: {counterparty}
Banks: {banks}
Amount: {amount}
Date: {date}
Transaction mode: {transaction_mode}
Account numbers: {account_numbers}
Countries: {countries}

Narrative:
{narrative}

Return only the summary."""


def clean(value: Any, fallback: str = "Not available") -> str:
    if value is None or pd.isna(value):
        return fallback

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback

    return re.sub(r"\s+", " ", text)


def build_user_prompt(row: pd.Series) -> str:
    return USER_PROMPT_TEMPLATE.format(
        report_id=clean(row.get("report_id")),
        report_type=clean(row.get("report_type")),
        customer=clean(row.get("entities_customer_names")),
        counterparty=clean(row.get("entities_counterparty_names")),
        banks=clean(row.get("entities_bank_names")),
        amount=clean(row.get("entities_amounts")),
        date=clean(row.get("entities_dates")),
        transaction_mode=clean(row.get("entities_transaction_modes")),
        account_numbers=clean(
            "; ".join(
                value
                for value in [
                    clean(row.get("from_account_number"), fallback=""),
                    clean(row.get("to_account_number"), fallback=""),
                ]
                if value
            )
        ),
        countries=clean(row.get("entities_countries")),
        narrative=clean(row.get("narrative")),
    )


def build_llama_prompt(row: pd.Series) -> str:
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
    return [
        ("system", SYSTEM_PROMPT),
        ("human", build_user_prompt(row)),
    ]


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
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


def select_rows(
    df: pd.DataFrame,
    limit: int | None,
    all_rows: bool,
    start: int,
) -> pd.DataFrame:
    selected = df.iloc[start:]
    if all_rows:
        return selected

    if limit is None:
        limit = 3

    return selected.head(limit)


def load_completed_report_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    existing = pd.read_csv(output_path, usecols=["report_id"])
    return set(existing["report_id"].dropna().astype(str))


def append_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_exists = output_path.exists()
    pd.DataFrame(rows).to_csv(
        output_path,
        mode="a" if output_exists else "w",
        header=not output_exists,
        index=False,
    )


def generate_summaries(
    df: pd.DataFrame,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    output_path: Path,
    batch_size: int,
    resume: bool,
) -> pd.DataFrame:
    completed_report_ids = load_completed_report_ids(output_path) if resume else set()
    pending = df[
        ~df["report_id"].astype(str).isin(completed_report_ids)
    ] if resume else df

    skipped_count = len(df) - len(pending)
    if skipped_count:
        print(f"Skipping {skipped_count} reports already in {output_path}")

    if pending.empty:
        print("No new summaries to generate.")
        return pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()

    llm = load_llm(model_id, max_new_tokens, temperature, top_p)
    rows: list[dict[str, Any]] = []
    generated_count = 0

    for _, row in pending.iterrows():
        report_id = clean(row.get("report_id"), fallback="")
        response = llm.invoke(build_messages(row))
        summary = clean(response.content, fallback="")

        rows.append(
            {
                "report_id": report_id,
                "report_type": row.get("report_type"),
                "model_id": model_id,
                "summary": summary,
            }
        )
        generated_count += 1

        print(f"Generated {report_id}")

        if len(rows) >= batch_size:
            append_rows(rows, output_path)
            rows.clear()

    append_rows(rows, output_path)
    print(f"Generated summaries this run: {generated_count}")
    return pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Track 6 STR summaries with a Hugging Face LLM."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--all", action="store_true", help="Summarize all rows.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based row offset.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="How many generated summaries to write per CSV append.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip report IDs already present in the output CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output before generation.",
    )
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
    selected = select_rows(df, args.limit, args.all, args.start)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    if args.dry_run:
        print(build_llama_prompt(selected.iloc[0]))
        print(f"\nDry run only. Rows selected: {len(selected)}")
        return

    if args.overwrite and args.output.exists():
        args.output.unlink()

    summaries = generate_summaries(
        selected,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_path=args.output,
        batch_size=args.batch_size,
        resume=args.resume,
    )

    print(f"Saved summaries: {args.output}")
    print(f"Total rows in output: {len(summaries)}")


if __name__ == "__main__":
    main()
