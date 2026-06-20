import pandas as pd
import numpy as np
import re
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer

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

def extractive_summary(narrative, ents, max_words=180):
    sentences = split_sentences(narrative)

    if len(sentences) <= 1:
        return narrative.strip()

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    scores = tfidf_matrix.sum(axis=1).A1

    terms = critical_entity_terms(ents)

    boosted_scores = []
    for i, s in enumerate(sentences):
        boost = 1.0
        for term in terms:
            if term in s.lower():
                boost += 0.5
        boosted_scores.append(scores[i] * boost)

    ranked = sorted(range(len(sentences)),
                    key=lambda i: boosted_scores[i],
                    reverse=True)

    selected = []
    word_count = 0

    for idx in ranked:
        words = sentences[idx].split()
        if word_count + len(words) <= max_words:
            selected.append((idx, sentences[idx]))
            word_count += len(words)

    selected = sorted(selected, key=lambda x: x[0])
    return " ".join([s for _, s in selected])

def type_a_summary(ents):
    return f"Customer {', '.join(ents.get('customer_names') or ['unknown'])} performed a transaction flagged for review."

def summarize_type_b(df_in):
    def _run(row):
        ents = parse_entities(row["entities_json"])
        return extractive_summary(str(row["narrative"]), ents)

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
