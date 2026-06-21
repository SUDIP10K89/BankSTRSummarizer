import pandas as pd
import numpy as np
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

def split_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)
    return [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]

def critical_entity_terms(ents):
    terms = []
    for key in ["customer_names", "counterparty_names", "bank_names",
                "amounts", "dates", "account_numbers"]:
        terms.extend(ents.get(key, []))
    return [t.lower() for t in terms if t]

def simple_extractive_summary(narrative, ents, max_words=180):
    """Simple extractive method using sentence relevance (no sklearn)."""
    sentences = split_sentences(narrative)

    if len(sentences) <= 1:
        # For minimal narratives, include structured entity data
        return _add_structured_entities(narrative.strip(), ents)

    # Score sentences by term frequency and entity matches
    terms = critical_entity_terms(ents)
    
    scored = []
    for i, s in enumerate(sentences):
        # Count matching terms
        score = sum(1 for term in terms if term in s.lower())
        # Also count words (prefer longer sentences)
        score += len(s.split()) / 50.0
        scored.append((i, s, score))
    
    # Select top sentences by score
    scored.sort(key=lambda x: x[2], reverse=True)
    
    selected = []
    word_count = 0
    for idx, sent, score in scored:
        words = sent.split()
        if word_count + len(words) <= max_words:
            selected.append((idx, sent))
            word_count += len(words)
    
    # Restore original order
    selected.sort(key=lambda x: x[0])
    summary = " ".join([s for _, s in selected])
    
    # Add missing critical entities
    return _add_missing_entities(summary, ents)


def _add_structured_entities(base_text, ents):
    """Add structured entities to a minimal text."""
    parts = [base_text]
    
    # Add transaction mode if available
    transaction_mode = ", ".join(ents.get('transaction_modes') or [])
    if transaction_mode and transaction_mode not in base_text.lower():
        parts.append(f"Transaction mode: {transaction_mode}.")
    
    # Add amount if available
    amount = ", ".join(ents.get('amounts') or [])
    if amount and amount not in base_text.lower():
        parts.append(f"Amount: {amount}.")
    
    # Add date if available
    date = ", ".join(ents.get('dates') or [])
    if date and date not in base_text.lower():
        parts.append(f"Date: {date}.")
    
    return " ".join(parts)


def _add_missing_entities(summary, ents):
    """Add critical missing entities to the summary."""
    summary_lower = summary.lower()
    
    # Check and add missing transaction mode
    transaction_mode = ", ".join(ents.get('transaction_modes') or [])
    if transaction_mode and transaction_mode.lower() not in summary_lower:
        # Insert transaction mode info at the end
        summary = summary.rstrip(".") + f" via {transaction_mode}."
    
    # Check and add missing amount
    amount = ", ".join(ents.get('amounts') or [])
    if amount and amount not in summary_lower:
        # Append amount info
        summary = summary.rstrip(".") + f" Amount: {amount}."
    
    # Check and add missing bank names
    banks = ents.get('bank_names', [])
    for bank in banks:
        if bank and bank not in summary:
            # Try to insert bank name naturally
            summary = summary.rstrip(".") + f" Institutions: {', '.join(banks)}."
            break
    
    return summary

def type_a_summary(ents):
    """Enhanced Type-A summary that includes structured entity data."""
    customer = ", ".join(ents.get('customer_names') or ['unknown'])
    counterparty = ", ".join(ents.get('counterparty_names') or [])
    banks = ", ".join(ents.get('bank_names') or [])
    amount = ", ".join(ents.get('amounts') or [])
    date = ", ".join(ents.get('dates') or [])
    transaction_mode = ", ".join(ents.get('transaction_modes') or [])
    
    # Build structured summary with all available entities
    parts = [f"Customer {customer} performed"]
    
    if transaction_mode:
        parts.append(f"a {transaction_mode.lower()}")
    else:
        parts.append("a transaction")
    
    if amount:
        parts.append(f"of {amount}")
    
    if date:
        parts.append(f"on {date}")
    
    if counterparty:
        parts.append(f"to {counterparty}")
    
    if banks:
        parts.append(f"at {banks}")
    
    parts.append("flagged for review.")
    
    return " ".join(parts)

def summarize_type_b(df_in):
    def _run(row):
        ents = parse_entities(row["entities_json"])
        return simple_extractive_summary(str(row["narrative"]), ents)

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

type_b_extractive = summarize_type_b(type_b)
type_a_summarized = summarize_type_a(type_a)

final_df = pd.concat([
    type_a_summarized[["report_id", "report_type", "summary"]],
    type_b_extractive[["report_id", "report_type", "summary"]],
])

os.makedirs("data", exist_ok=True)
final_df.to_csv("data/summary_extractive.csv", index=False)

print("Saved: data/summary_extractive.csv")
