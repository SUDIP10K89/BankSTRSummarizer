import pandas as pd
import re
import json
import os

df = pd.read_csv("data/track6_reports_with_entities.csv")

type_a = df[df['report_type'] == 'Type-A'].copy()
type_b = df[df['report_type'] == 'Type-B'].copy()

def parse_entities(entities_json_str):
    try:
        d = json.loads(entities_json_str)
    except:
        d = {}
    fields = ["account_numbers", "amounts", "bank_names", "counterparty_names",
              "countries", "customer_names", "dates", "transaction_modes"]
    return {f: d.get(f, []) or [] for f in fields}

def format_list(items, default):
    return ", ".join(items) if items else default

def format_amount(amounts):
    if not amounts:
        return "an undisclosed amount"
    return "; ".join(amounts)

def format_date(dates):
    if not dates:
        return "an unspecified date"
    return f"{dates[0]} to {dates[-1]}" if len(dates) > 1 else dates[0]

def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

def extract_context(narrative):
    keywords = ["pattern", "suspicious", "threshold", "velocity"]
    sentences = split_sentences(narrative)
    return " ".join([s for s in sentences if any(k in s.lower() for k in keywords)])

def hybrid_summary(narrative, ents):
    customers = ents.get("customer_names")
    counterparties = ents.get("counterparty_names")
    banks = ents.get("bank_names")
    amounts = ents.get("amounts")
    dates = ents.get("dates")
    modes = ents.get("transaction_modes")

    context = extract_context(narrative)

    summary = (
        f"Customer {format_list(customers, 'unknown')} at {format_list(banks, 'unknown bank')} "
        f"conducted {format_list(modes, 'a')} transaction totaling {format_amount(amounts)} "
        f"on {format_date(dates)} to {format_list(counterparties, 'unknown party')}. "
        f"{context}"
    )
    return summary

def type_a_summary(ents):
    return f"Customer {', '.join(ents.get('customer_names') or ['unknown'])} performed a transaction flagged for review."

def summarize_type_b(df_in):
    def _run(row):
        ents = parse_entities(row["entities_json"])
        return hybrid_summary(str(row["narrative"]), ents)

    out = df_in.copy()
    out["summary"] = out.apply(_run, axis=1)
    return out

def summarize_type_a(df_in):
    def _run(row):
        ents = parse_entities(row["entities_json"])
        return type_a_summary(ents)

    out = df_in.copy()
    out["summary"] = out.apply(_run, axis=1)
    return out

type_b_hybrid = summarize_type_b(type_b)
type_a_summarized = summarize_type_a(type_a)

final_df = pd.concat([
    type_a_summarized[["report_id", "report_type", "summary"]],
    type_b_hybrid[["report_id", "report_type", "summary"]],
])

os.makedirs("data", exist_ok=True)
final_df.to_csv("data/summary_hybrid.csv", index=False)

print("Saved: data/summary_hybrid.csv")
