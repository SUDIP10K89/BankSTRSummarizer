

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "data" / "track6_reports_with_entities.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "track6_llm_summaries_local.csv"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  


SYSTEM_PROMPT = """You are a financial crime analyst writing neutral STR summaries.
Preserve facts exactly. Do not invent names, amounts, dates, banks, locations, or legal conclusions."""


USER_PROMPT_TEMPLATE = """Write a concise Suspicious Transaction Report summary in 100-200 words.

Rules:
- CRITICAL: Use the exact date format, amounts, transaction mode, and entity names provided in the structured facts below. Copy them verbatim into your summary.
- CRITICAL: Always include ALL bank/institution names provided in the "Banks" field. Do not omit secondary banks even if it requires more words.
- Include every available customer name, counterparty, bank/institution name, amount, date, transaction mode, and account number.
- When mentioning the date, use the exact format provided: {date}
- When mentioning the transaction mode, use the exact term provided: {transaction_mode}
- When listing banks, include all of them. If multiple banks are listed (e.g., "Bank A; Bank B"), mention both: "involving Bank A and Bank B" or "through Bank A and Bank B".
- If a key fact is not available, say it is not available. If you intentionally exclude a key fact, state the reason briefly.
- Use only the structured facts and narrative below. Do not add facts from outside this input.
- If the narrative is minimal, summarize using the structured facts only.
- Do not say the customer committed a crime.
- Keep a neutral analyst-facing tone that can be read in under 30 seconds.
- The output must be one paragraph of 100-200 words.

Structured facts (use exact values):
Report ID: {report_id}
Report type: {report_type}
Customer: {customer}
Counterparty: {counterparty}
Banks (list all): {banks}
Amount: {amount}
Date (use this format): {date}
Transaction mode (use this exact term): {transaction_mode}
Account numbers: {account_numbers}
Countries: {countries}

Narrative:
{narrative}

Return only the summary. Remember to:
1. Use exact date format, amounts, and transaction mode from structured facts
2. Include ALL banks/institutions from the Banks field
3. Keep output 100-200 words"""


def clean(value: Any, fallback: str = "Not available") -> str:
    """Return clean display text for prompt fields."""
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


def build_messages(row: pd.Series) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row)},
    ]


class LocalLLM:

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        device: str = "cuda",  # cuda, cpu, mps
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: install transformers and torch to use this script."
            ) from exc

        print(f"Loading {self.model_id} on device {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Use fp16 for memory efficiency on 6GB GPU
        import torch
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Model loaded successfully on {self.device}.")

    def invoke(self, messages: list[dict[str, str]]) -> str:
        prompt_text = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt_text += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt_text += f"User: {msg['content']}\n\nAssistant: "

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Assistant: " in response:
            response = response.split("Assistant: ", 1)[1]

        return response.strip()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()

    def cleanup(self):
        """Free GPU memory."""
        try:
            import torch

            if self.model:
                self.model.to("cpu")
            torch.cuda.empty_cache()
        except Exception:
            pass


def select_rows(
    df: pd.DataFrame,
    limit: int | None,
    all_rows: bool,
    start: int,
) -> pd.DataFrame:
    """Pick rows to summarize."""
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
    """Append generated rows to the output CSV."""
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
    device: str = "cuda",
) -> pd.DataFrame:
    """Generate summaries for selected rows using transformers."""
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

    with LocalLLM(
        model_id,
        max_new_tokens,
        temperature,
        top_p,
        device=device,
    ) as llm:
        rows: list[dict[str, Any]] = []
        generated_count = 0

        for idx, (_, row) in enumerate(pending.iterrows(), start=1):
            report_id = clean(row.get("report_id"), fallback="")
            messages = build_messages(row)

            try:
                response = llm.invoke(messages)
                summary = clean(response, fallback="")
            except Exception as e:
                print(f"Error generating {report_id}: {e}")
                summary = f"Error: {str(e)}"

            rows.append(
                {
                    "report_id": report_id,
                    "report_type": row.get("report_type"),
                    "model_id": model_id,
                    "summary": summary,
                }
            )
            generated_count += 1

            print(f"[{idx}/{len(pending)}] Generated {report_id}")

            if len(rows) >= batch_size:
                append_rows(rows, output_path)
                rows.clear()

        append_rows(rows, output_path)
        print(f"Generated summaries this run: {generated_count}")

    return pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Track 6 STR summaries with local Hugging Face LLM."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID (full path, e.g. meta-llama/Llama-2-7b-chat-hf).",
    )
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
        "--device",
        choices=["cuda", "cpu", "mps"],
        default="cuda",
        help="Device to run model on (cuda/cpu/mps).",
    )
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
        user_prompt = build_user_prompt(selected.iloc[0])
        print(f"System:\n{SYSTEM_PROMPT}\n")
        print(f"User:\n{user_prompt}\n")
        print(f"Dry run only. Rows selected: {len(selected)}")
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
        device=args.device,
    )

    print(f"Saved summaries: {args.output}")
    print(f"Total rows in output: {len(summaries)}")


if __name__ == "__main__":
    main()
