"""
Fine-tune facebook/bart-base on AML/SAR report summarization.

Dataset columns used:
  - input : narrative  (the long compliance text)
  - target: generated summary derived from structured fields

For Type-A rows (short narrative), we auto-generate a structured summary.
For Type-B rows, we compress the narrative into a 1-2 sentence summary.

Usage:
    pip install transformers datasets torch accelerate sentencepiece
    python finetune_bart_aml.py
"""

import json
import re
import os
import csv
import io
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import numpy as np

# ─────────────────────────────────────────────
# 1.  RAW DATA  (paste your CSV rows here or load from file)
# ─────────────────────────────────────────────

RAW_CSV = """\
report_id,report_type,narrative,from_institution,from_account_name,to_institution,to_account_name,amount_local,currency_code,transaction_mode,transaction_location,from_country,to_country,entities_customer_names,entities_counterparty_names
RPT-2026-000001,Type-A,Suspicious transaction observed.,PCBL,John Jensen,SBL,Jeremy Martinez,535368.64,NPR,Cash Deposit transfer,Ghorahi,GB,GB,John Jensen,Jeremy Martinez
RPT-2026-000002,Type-B,"During the branch's periodic transaction-monitoring review, the compliance desk at ADBL examined the account held by Sally Dominguez after an internal threshold alert was triggered. Between 2022-10-07 and 2022-10-07, the customer conducted 1 transaction(s) — predominantly cash withdrawals — amounting to approximately NPR 21,807.",ADBL,Sally Dominguez,HBL,Henry King,21807.13,NPR,Cash Withdrawal transfer,Birgunj,GB,IN,Sally Dominguez,Henry King
RPT-2026-000003,Type-B,"During the branch's periodic transaction-monitoring review, the compliance desk at KUMARI examined the account held by Cook, Flores and Ray after an internal threshold alert was triggered. Between 2022-10-07 and 2022-10-07, the customer conducted 1 transaction(s) — predominantly cross-border wires — amounting to approximately NPR 1,214,020.",KUMARI,"Cook, Flores and Ray",SBL,Chad Mccarty,1214019.81,NPR,Cross-border transfer,Hetauda,GB,MX,"Cook, Flores and Ray",Chad Mccarty
RPT-2026-000004,Type-B,"During the branch's periodic transaction-monitoring review, the compliance desk at PRABHU examined the account held by Robert King after an internal threshold alert was triggered. Between 2022-10-07 and 2022-10-07, the customer conducted 1 transaction(s) — predominantly cross-border wires — amounting to approximately NPR 813,936.",PRABHU,Robert King,LAXMI,Sarah English,813935.97,NPR,Cross-border transfer,Birgunj,GB,NG,Robert King,Sarah English
RPT-2026-000005,Type-A,Suspicious transaction observed.,MEGA,Porter LLC,MEGA,Terrence Davis,1610735.12,NPR,Debit card transfer,Dharan,GB,GB,Porter LLC,Terrence Davis
RPT-2026-000006,Type-B,"During the branch's periodic transaction-monitoring review, the compliance desk at PRABHU examined the account held by Robert Gould after an internal threshold alert was triggered. Between 2022-10-07 and 2022-10-07, the customer conducted 1 transaction(s) — predominantly cross-border wires — amounting to approximately NPR 668,846.",PRABHU,Robert Gould,KUMARI,Kathy Haynes,668845.78,NPR,Cross-border transfer,Nepalgunj,GB,US,Robert Gould,Kathy Haynes
RPT-2026-000007,Type-B,"During the branch's periodic transaction-monitoring review, the compliance desk at NIMB examined the account held by Kimberly Aguirre after an internal threshold alert was triggered. Between 2022-10-07 and 2022-10-07, the customer conducted 1 transaction(s) — predominantly cross-border wires — amounting to approximately NPR 10,861.",NIMB,Kimberly Aguirre,PCBL,Oconnor and Sons,10860.8,NPR,Cross-border transfer,Ghorahi,JP,GB,Kimberly Aguirre,Oconnor and Sons
RPT-2026-000008,Type-B,"During the branch's periodic transaction-monitoring review, the compliance desk at CBL examined the account held by Xavier Bush after an internal threshold alert was triggered. Between 2022-10-07 and 2022-10-07, the customer conducted 2 transaction(s) — predominantly card payments — amounting to approximately NPR 3,002,367.",CBL,Xavier Bush,ADBL,C. Cooke,2845747.68,NPR,Credit card transfer,Bhairahawa,GB,GB,Xavier Bush,C. Cooke
RPT-2026-000009,Type-A,Suspicious transaction observed.,PRIME,Michael Tanner,SUNRISE,Debra Michael,862500.21,NPR,Cash Deposit transfer,Janakpur,GB,GB,Michael Tanner,Debra Michael
RPT-2026-000010,Type-B,"During the branch's periodic transaction-monitoring review, the compliance desk at NCC examined the account held by Michael Ruiz after an internal threshold alert was triggered. Between 2022-10-07 and 2022-10-07, the customer conducted 1 transaction(s) — predominantly card payments — amounting to approximately NPR 2,161,766.",NCC,Michael Ruiz,CITIZENS,Scott Jones,2161766.07,NPR,Debit card transfer,Ghorahi,GB,GB,Michael Ruiz,Scott Jones
"""

# ─────────────────────────────────────────────
# 2.  SUMMARY GENERATION  (rule-based labels)
# ─────────────────────────────────────────────

def build_summary(row: dict) -> str:
    """
    Create a concise 1-2 sentence reference summary from structured fields.
    This acts as the training TARGET.
    """
    customer   = row["from_account_name"]
    cpty       = row["to_account_name"]
    amount     = row["amount_local"]
    currency   = row["currency_code"]
    mode       = row["transaction_mode"]
    location   = row["transaction_location"]
    from_bank  = row["from_institution"]
    to_bank    = row["to_institution"]
    from_cty   = row["from_country"]
    to_cty     = row["to_country"]

    cross = "" if from_cty == to_cty else f" (cross-border: {from_cty}→{to_cty})"
    summary = (
        f"Suspicious {mode} of {currency} {float(amount):,.2f} "
        f"from {customer} at {from_bank} to {cpty} at {to_bank}, "
        f"executed in {location}{cross}. "
        f"Flagged for possible money-laundering indicators including "
        f"threshold avoidance and rapid succession of transfers."
    )
    return summary


# ─────────────────────────────────────────────
# 3.  DATASET
# ─────────────────────────────────────────────

def load_records() -> list[dict]:
    reader = csv.DictReader(io.StringIO(RAW_CSV))
    records = []
    for row in reader:
        src = row["narrative"].strip()
        tgt = build_summary(row)
        records.append({"input": src, "target": tgt, "report_id": row["report_id"]})
    return records


class AMLDataset(Dataset):
    def __init__(self, records, tokenizer, max_src=512, max_tgt=128):
        self.records   = records
        self.tokenizer = tokenizer
        self.max_src   = max_src
        self.max_tgt   = max_tgt

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        model_inputs = self.tokenizer(
            rec["input"],
            max_length=self.max_src,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            rec["target"],
            max_length=self.max_tgt,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label_ids = labels["input_ids"].squeeze()
        # Replace padding token id with -100 so loss ignores them
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels":         label_ids,
        }


# ─────────────────────────────────────────────
# 4.  ROUGE METRIC
# ─────────────────────────────────────────────

def compute_rouge(pred, tokenizer):
    """Simple token-overlap ROUGE-L (no external library needed)."""
    preds  = pred.predictions
    labels = pred.label_ids

    # Replace -100 back to pad id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    def lcs(a, b):
        a, b = a.split(), b.split()
        m, n = len(a), len(b)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    scores = []
    for pred_txt, label_txt in zip(decoded_preds, decoded_labels):
        p_len = len(pred_txt.split())
        l_len = len(label_txt.split())
        if p_len == 0 or l_len == 0:
            scores.append(0.0)
            continue
        lcs_len = lcs(pred_txt, label_txt)
        prec = lcs_len / p_len
        rec  = lcs_len / l_len
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        scores.append(f1)

    return {"rouge_l": round(float(np.mean(scores)), 4)}


# ─────────────────────────────────────────────
# 5.  MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def main():
    MODEL_NAME = "facebook/bart-base"
    OUTPUT_DIR = "./aml_bart_finetuned"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading tokenizer and model …")
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model     = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    records = load_records()
    print(f"\n{len(records)} records loaded.")
    for r in records:
        print(f"  [{r['report_id']}]")
        print(f"    SRC: {r['input'][:80]}…")
        print(f"    TGT: {r['target'][:80]}…\n")

    # With only 10 samples: use all for train, 2 for eval (overlap is acceptable
    # for a demo; in production you'd split properly).
    train_records = records
    eval_records  = records[:2]

    train_dataset = AMLDataset(train_records, tokenizer)
    eval_dataset  = AMLDataset(eval_records,  tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = 10,          # more epochs for small dataset
        per_device_train_batch_size = 2,
        per_device_eval_batch_size  = 2,
        warmup_steps                = 5,
        weight_decay                = 0.01,
        logging_steps               = 5,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "rouge_l",
        predict_with_generate       = True,
        generation_max_length       = 128,
        fp16                        = torch.cuda.is_available(),
        report_to                   = "none",
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = eval_dataset,
        data_collator   = data_collator,
        compute_metrics = lambda p: compute_rouge(p, tokenizer),
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\nStarting fine-tuning …")
    trainer.train()

    print("\nSaving model …")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

    # ── Inference demo ──────────────────────────────────────────────────
    print("\n─── INFERENCE DEMO ───")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_inputs = [
        "Suspicious transaction observed.",
        (
            "During the branch's periodic transaction-monitoring review, "
            "the compliance desk at NCC examined the account held by Michael Ruiz "
            "after an internal threshold alert was triggered. "
            "The customer conducted 1 transaction amounting to approximately NPR 2,161,766 "
            "via Debit card transfer, routed to Scott Jones at CITIZENS."
        ),
    ]

    for txt in test_inputs:
        enc = tokenizer(txt, return_tensors="pt", max_length=512, truncation=True).to(device)
        out = model.generate(**enc, max_length=128, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\nINPUT : {txt[:100]}…")
        print(f"OUTPUT: {summary}")


if __name__ == "__main__":
    main()
