# BankSTRSummarizer

**Automated summarization pipeline for Suspicious Transaction Reports (STRs)**

Repository: [github.com/SUDIP10K89/BankSTRSummarizer](https://github.com/SUDIP10K89/BankSTRSummarizer)

---

## 1. Overview

BankSTRSummarizer takes raw, XML-formatted Suspicious Transaction Reports (the kind banks file with financial intelligence units) and turns them into short, fact-preserving, analyst-readable summaries (100–200 words). It was built for **Track 6** of a hackathon/challenge focused on automated STR summarization for compliance teams.

The pipeline does four things end-to-end:

1. **Parses** raw STR XML reports into a structured dataset.
2. **Extracts entities** (names, banks, amounts, dates, transaction modes, account numbers) from both structured fields and free-text narratives (via spaCy NER).
3. **Generates summaries** using four different strategies — extractive, rule-based hybrid, cloud LLM (Hugging Face), and local LLM — so they can be compared.
4. **Evaluates** every summary against the source facts for length, entity preservation, and (optionally) ROUGE-L / BERTScore.

This makes it possible to answer: *"Which summarization approach best compresses an STR while keeping every legally/operationally relevant fact intact?"*

---

## 2. Why This Exists

STR narratives vary wildly in quality:

- **Type-A reports**: minimal narrative (often just the placeholder text *"Suspicious transaction observed."*) — almost no free-text detail, everything must be reconstructed from structured fields.
- **Type-B reports**: rich, multi-sentence narratives describing the suspicious pattern, KYC context, and compliance officer's reasoning.

A single summarization strategy doesn't work well for both. This project classifies each report by type and applies the right strategy, then measures whether critical facts (customer, counterparty, banks, amount, date, transaction mode, account numbers) survive the compression.

---

## 3. Repository Structure

```
BankSTRSummarizer/
├── reports/                          # 276 raw STR XML files (report_000001.xml ... report_000276.xml)
├── data/                             # Generated datasets, summaries, and evaluation outputs
│   ├── track6_reports_dataset.csv            # Parsed XML → tabular dataset
│   ├── track6_reports_with_entities.csv      # + extracted entities
│   ├── track6_llm_summaries.csv              # Cloud LLM summaries
│   ├── track6_llm_summaries_local.csv        # Local LLM summaries
│   ├── summary_extractive.csv                # Extractive summaries
│   ├── summary_hybrid.csv                    # Rule-based hybrid summaries
│   ├── track6_evaluation_results*.csv        # Per-report evaluation results (one per method)
│   ├── track6_evaluation_summary*.json       # Aggregate evaluation metrics (one per method)
│   ├── track6_model_comparison_results.csv   # Side-by-side model comparison
│   ├── track6_model_comparison_summary.json
│   └── track6_demo_examples.md               # Hand-picked example outputs for presentation
├── data_pipeline.py                  # Step 1: XML → CSV
├── entity_extraction.py              # Step 2: entity extraction (structured + spaCy NER)
├── llm_summarizer.py                 # Step 3a: cloud LLM summarization (Hugging Face Inference Endpoint)
├── llm_summarizer_local.py           # Step 3b: local LLM summarization (transformers, on-device)
├── extractive_summary.py             # Step 3c: extractive summarization (sentence scoring, no LLM)
├── hybrid_summary.py                 # Step 3d: rule-based template summarization
├── evaluation.py                     # Step 4: fact-preservation + length + ROUGE-L/BERTScore evaluation
├── pipeline.py                       # Orchestrator — runs the full pipeline end-to-end
├── eda.ipynb                         # Exploratory data analysis notebook
├── track6_model_comparison.ipynb     # Notebook comparing all four summarization methods
└── requirements.txt
```

**Languages:** ~94% Jupyter Notebook, ~6% Python (per GitHub's language breakdown).

---

## 4. Pipeline Walkthrough

### Step 1 — `data_pipeline.py`: Parse raw reports

Reads every `report_*.xml` file in `reports/`, extracts fields via `xml.etree.ElementTree`, and writes a flat CSV.

Key fields captured per report:

| Field | Source |
|---|---|
| `report_id`, `entity_reference`, `submission_date`, `currency_code` | Report root |
| `narrative`, `narrative_length`, `report_type` | `<reason>` text; classified as **Type-A** (empty/placeholder) or **Type-B** (substantive) |
| `transaction_number`, `amount_local`, `date_transaction`, `transaction_location`, `transaction_mode` | `<transaction>` block |
| `from_institution`, `from_account_name`, `from_account_number`, `from_country` | Sender details (`t_from_my_client`) |
| `to_institution`, `to_account_name`, `to_account_number`, `to_country` | Recipient details (`t_to`) |

**Run:**
```bash
python data_pipeline.py --reports-dir reports --output data/track6_reports_dataset.csv
```
**Output:** `data/track6_reports_dataset.csv`

---

### Step 2 — `entity_extraction.py`: Enrich with entities

Combines two sources of entities per report:

1. **Structured fields** already in the CSV (customer/counterparty names, banks, amount+currency, dates, transaction modes, countries, account numbers).
2. **spaCy NER** (`en_core_web_sm` by default) run over the free-text `narrative`, extracting `PERSON`, `ORG`, `MONEY`, `DATE`, and `GPE` (location) entities. If spaCy or the model isn't installed, the script gracefully falls back to structured-fields-only.

Outputs both a combined `entities_json` column (full structured bundle) and flattened semicolon-joined columns (`entities_customer_names`, `entities_bank_names`, etc.) for easy downstream use.

**Run:**
```bash
python entity_extraction.py --input data/track6_reports_dataset.csv --output data/track6_reports_with_entities.csv
# Add --no-spacy to skip NER and use structured fields only
```
**Output:** `data/track6_reports_with_entities.csv`

---

### Step 3 — Four summarization strategies

All four take `track6_reports_with_entities.csv` as input and produce a `summary` per `report_id`.

#### 3a. Extractive — `extractive_summary.py`
- For **Type-B** (rich narrative): splits the narrative into sentences, scores each by how many critical entity terms it contains plus a length bonus, greedily selects top sentences up to ~180 words while preserving original order, then appends any still-missing critical entities (transaction mode, amount, bank names).
- For **Type-A** (minimal narrative): builds a templated sentence directly from structured entities (customer, mode, amount, date, counterparty, banks).
- No ML model required — pure rule/heuristic based.
- **Output:** `data/summary_extractive.csv`

#### 3b. Hybrid — `hybrid_summary.py`
- Rule-based template summary for both report types, but pulls additional **context sentences** out of the narrative using suspicious-activity keywords (`pattern`, `suspicious`, `threshold`, `velocity`) and appends them after the structured fact sentence.
- **Output:** `data/summary_hybrid.csv`

#### 3c. Cloud LLM — `llm_summarizer.py`
- Uses `langchain-huggingface` to call a Hugging Face Inference Endpoint (default model: `meta-llama/Llama-3.1-70B-Instruct`).
- Sends a system prompt instructing the model to act as a neutral financial-crime analyst, never invent facts, and produce a 100–200 word paragraph.
- Supports resuming (`--resume`), batch writing, dry runs (`--dry-run`), and row limits/offsets for incremental generation.
- Requires a Hugging Face token via `HUGGINGFACEHUB_API_TOKEN` or `HF_TOKEN` (loaded from a local `.env` file if present).
- **Output:** `data/track6_llm_summaries.csv`

**Run:**
```bash
python llm_summarizer.py --resume --limit 25
```

#### 3d. Local LLM — `llm_summarizer_local.py`
- Same prompt/task as the cloud version, but runs a model fully on-device via `transformers` (default: `Qwen/Qwen2.5-3B-Instruct`, fp16, GPU by default with `--device cuda/cpu/mps`).
- Stricter prompt: explicitly instructs the model to copy structured field values (dates, amounts, transaction modes) **verbatim** and to never drop secondary banks.
- **Output:** `data/track6_llm_summaries_local.csv`

**Run:**
```bash
python llm_summarizer_local.py --resume --limit 25 --device cuda
```

---

### Step 4 — `evaluation.py`: Score every summary

For each generated summary, checks:

| Check | Meaning |
|---|---|
| `word_count` / `length_ok` | Is the summary 100–200 words? |
| `has_customer`, `has_counterparty`, `has_banks` | Are these names present in the summary text? |
| `has_amount` | Are expected amounts present (numeric match with tolerance scaled to magnitude)? |
| `has_date` | Are expected dates present (handles ISO / US / EU date formats)? |
| `has_transaction_mode` | Is the transaction mode present (keyword-normalized, e.g. "wire", "cash", "cross-border")? |
| `has_account_numbers` | Are account numbers present verbatim? |
| `critical_entity_preservation_ok` | All of: customer, counterparty, banks, amount, date, transaction mode |
| `strict_entity_preservation_ok` | Critical + account numbers |
| `overall_ok` | `length_ok` AND `critical_entity_preservation_ok` |
| `rouge_l_f1`, `bertscore_f1` | Optional, only computed if a gold-reference summary column is supplied |

Produces a per-report CSV plus an aggregate JSON summary (pass rates, average word count, etc.).

**Run:**
```bash
python evaluation.py --summaries data/track6_llm_summaries.csv \
                      --output data/track6_evaluation_results.csv \
                      --summary-output data/track6_evaluation_summary.json
```

**Sample aggregate result** (cloud LLM, 25 reports):

| Metric | Value |
|---|---|
| Average word count | 114.4 |
| Length pass rate | 80% |
| Critical entity preservation rate | 100% |
| Strict entity preservation rate | 100% |
| Overall pass rate | 80% |

---

### Orchestration — `pipeline.py`

Runs every step above in sequence (parse → extract entities → cloud LLM summarize → evaluate → local LLM summarize → evaluate → extractive → evaluate → hybrid → evaluate), stopping early if any script fails.

**Run the full pipeline:**
```bash
python pipeline.py
```

---

## 5. Notebooks

- **`eda.ipynb`** — exploratory data analysis over the parsed report dataset (distribution of report types, narrative lengths, transaction amounts, etc.).
- **`track6_model_comparison.ipynb`** — side-by-side comparison of the extractive, hybrid, cloud-LLM, and local-LLM summaries, feeding into `track6_model_comparison_results.csv` / `track6_model_comparison_summary.json`.

---

## 6. Setup

```bash
git clone https://github.com/SUDIP10K89/BankSTRSummarizer.git
cd BankSTRSummarizer
pip install -r requirements.txt

# Optional, for narrative NER in entity_extraction.py:
python -m spacy download en_core_web_sm
```

**`requirements.txt`:**
```
pandas
numpy
scikit-learn
langchain-huggingface
huggingface-hub
transformers
torch
accelerate
sentencepiece
bert-score
spacy
jupyter
matplotlib
seaborn
```

### Credentials (for cloud LLM summarization only)

Create a `.env` file in the project root:
```
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

---

## 7. End-to-End Example

```bash
# 1. Parse raw reports
python data_pipeline.py

# 2. Extract entities
python entity_extraction.py

# 3a. Rule-based summaries (fast, no model needed)
python extractive_summary.py
python hybrid_summary.py

# 3b. LLM summaries (needs HF token or local GPU)
python llm_summarizer.py --limit 25
python llm_summarizer_local.py --limit 25 --device cuda

# 4. Evaluate each method
python evaluation.py --summaries data/summary_extractive.csv \
                      --output data/track6_evaluation_results_extractive.csv \
                      --summary-output data/track6_evaluation_summary_extractive.json
```

Or simply:
```bash
python pipeline.py
```

---

## 8. Sample Output

From `data/track6_demo_examples.md` (Type-A report, minimal narrative reconstructed entirely from structured fields):

> *A Suspicious Transaction Report (RPT-2026-000001) of type Type-A has been filed regarding a transaction between customer John Jensen and counterparty Jeremy Martinez. The transaction occurred on 2022-10-07, involving a cash deposit transfer of NPR 535,368.64. The transaction involved account numbers ... at banks PCBL and SBL, respectively...*

**Validation:** PASS — length OK, all critical and strict facts preserved.

---

## 9. Data Privacy Note

The `reports/` directory and all derived CSVs contain **synthetic/sample data** (fictional names, account numbers, and transaction details) used for development and evaluation purposes — not real customer or bank data.

---

## 10. Status

This is a hackathon/challenge project (Track 6: STR summarization). There is currently no license file, no CI pipeline, and no published releases — treat it as a working prototype rather than a production-ready package.