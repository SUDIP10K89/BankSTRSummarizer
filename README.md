# BankSTRSummarizer — AI-Powered STR Summarization

**Track 6 — AI-Powered Analysis & Reporting**  
Hackathon project that automatically summarizes Suspicious Transaction Reports (STRs) from financial institutions into concise, factually faithful 100–200 word summaries.

## Table of Contents

- [Overview](#overview)
- [Architecture & Data Flow](#architecture--data-flow)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)

## Overview

Financial institutions file Suspicious Transaction Reports (STRs) when they detect potentially suspicious activity. Each STR contains a 1,000–8,000 character narrative written by a reporting officer. Analysts receiving hundreds of these reports need concise, accurate summaries to triage quickly.

This project implements and compares **four summarization approaches**:

| Approach | Model / Method | Type |
|---|---|---|
| **LLM API** | Llama-3.1-70B-Instruct via Hugging Face | Abstractive (cloud) |
| **Local LLM** | Qwen2.5-3B-Instruct via Transformers | Abstractive (local) |
| **Extractive** | TF-IDF + entity-boosted sentence ranking | Extractive |
| **Hybrid** | Template-based with structured entities | Template + extractive |

All approaches are evaluated on **factual faithfulness** (entity preservation) and **length compliance** (100–200 words).

## Architecture & Data Flow

```
reports/report_*.xml  (276 raw STR XML files)
        |
        v
  data_pipeline.py     ->  data/track6_reports_dataset.csv
        |
        v
  entity_extraction.py ->  data/track6_reports_with_entities.csv
        |
        +----------+----------+-------------+-------------+
        |          |          |             |             |
        v          v          v             v             v
  llm_       llm_       extractive_   hybrid_
  summarizer  summarizer  summary.py    summary.py
  .py (API)  _local.py   |             |
        |          |          |             |
        v          v          v             v
  track6_    track6_    summary_      summary_
  llm_       llm_       extractive    hybrid.csv
  summaries  summaries  .csv
  .csv       _local.csv
        |          |          |             |
        +----------+----------+-------------+
        |
        v
  evaluation.py          ->  track6_evaluation_results*.csv
                             track6_evaluation_summary*.json
        |
        v
  track6_model_comparison.ipynb  ->  comparison results + demo examples
```

## Setup

### Prerequisites

- Python 3.10+
- [Hugging Face API token](https://huggingface.co/settings/tokens) (for the cloud LLM summarizer)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd BankSTRSummarizer

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Download spaCy model for NER-based entity enrichment
python -m spacy download en_core_web_sm

# Configure Hugging Face API token
echo "HUGGINGFACEHUB_API_TOKEN=hf_your_token_here" > .env
```

## Usage

### 1. Data Pipeline — Parse XML reports

```bash
python data_pipeline.py
```

Parses all 276 XML files in `reports/` and outputs `data/track6_reports_dataset.csv`.

### 2. Entity Extraction — Enrich with structured entities

```bash
python entity_extraction.py
```

Reads the parsed CSV and extracts customer names, counterparties, banks, amounts, dates, transaction modes, and countries. Optionally uses spaCy for NER. Outputs `data/track6_reports_with_entities.csv`.

### 3. Generate Summaries

Pick one or more summarizers:

```bash
# LLM API (cloud — requires HF token, defaults to 3 reports)
python llm_summarizer.py --limit 10

# Local LLM (runs on GPU/CPU — 3B parameter model)
python llm_summarizer_local.py --limit 10 --device cuda

# Extractive baseline (all 276 reports)
python extractive_summary.py

# Hybrid template baseline (all 276 reports)
python hybrid_summary.py
```

Common flags for LLM summarizers:
- `--all` — Process all reports
- `--resume` — Skip already-processed report IDs
- `--dry-run` — Print prompts without calling the model
- `--limit N` — Process only N reports
- `--batch-size N` — Flush to CSV every N reports

### 4. Evaluate Summaries

```bash
python evaluation.py
```

Evaluates length compliance and entity preservation. Defaults to evaluating the LLM API summaries. Use `--summaries-path` and `--output-prefix` for other models:

```bash
python evaluation.py \
    --summaries-path data/summary_extractive.csv \
    --output-prefix _extractive
```

### 5. Run Everything End-to-End

```bash
python pipeline.py
```

Orchestrates all steps in sequence (data pipeline -> entity extraction -> all four summarizers -> evaluations). Stops on first failure.

### 6. Exploratory Data Analysis & Model Comparison

```bash
jupyter notebook eda.ipynb
jupyter notebook track6_model_comparison.ipynb
```

## Results

### Model Comparison Summary

| Metric | LLM API | Local LLM | Extractive | Hybrid |
|---|---|---|---|---|
| **Summaries** | 10 | 10 | 276 | 276 |
| **Avg word count** | 117.7 | 101.0 | 121.1 | 75.3 |
| **Length OK** (100-200 words) | **80.0%** | 60.0% | 63.8% | 63.8% |
| **Critical entities OK** | **100.0%** | 0.0% | 0.0% | 63.8% |
| **Strict entities OK** (+ account #) | **100.0%** | 0.0% | 0.0% | 0.0% |
| **Overall pass rate** | **80.0%** | 0.0% | 0.0% | 63.8% |

**Winner: LLM API (Llama-3.1-70B-Instruct)** — 80% overall pass rate with perfect entity preservation.  
The 3B-parameter local model struggled with instruction-following at this task complexity. The extractive baseline failed on entity preservation because key facts don't always appear verbatim in narrative sentences. The hybrid template performed decently on entity coverage but consistently produced summaries below the 100-word minimum.

## Project Structure

```
BankSTRSummarizer/
+-- data/
|   +-- track6_reports_dataset.csv        # Parsed XML -> structured CSV
|   +-- track6_reports_with_entities.csv  # Entity-enriched dataset
|   +-- track6_llm_summaries.csv          # LLM API summaries
|   +-- track6_llm_summaries_local.csv    # Local LLM summaries
|   +-- summary_extractive.csv            # Extractive baseline summaries
|   +-- summary_hybrid.csv                # Hybrid template summaries
|   +-- track6_evaluation_results*.csv    # Per-report evaluation scores
|   +-- track6_evaluation_summary*.json   # Aggregate evaluation metrics
|   +-- track6_model_comparison_results.csv
|   +-- track6_model_comparison_summary.json
|   +-- track6_demo_examples.md           # Presentation-ready demo
+-- reports/                              # 276 raw STR XML files
|   +-- report_000001.xml ... report_000276.xml
+-- data_pipeline.py                      # Task 1.1: XML -> CSV
+-- entity_extraction.py                  # Task 1.2: Entity enrichment
+-- evaluation.py                         # Task 1.3: Summary evaluation
+-- llm_summarizer.py                     # Task 3.2: Cloud LLM summarizer
+-- llm_summarizer_local.py               # Local LLM summarizer
+-- extractive_summary.py                 # Extractive baseline
+-- hybrid_summary.py                     # Hybrid template baseline
+-- pipeline.py                           # End-to-end orchestration
+-- eda.ipynb                             # Exploratory data analysis
+-- track6_model_comparison.ipynb         # Model comparison & results
+-- explore_xml.py                        # (superseded by data_pipeline.py)
+-- requirements.txt
+-- .env                                  # HF API token (gitignored)
+-- .gitignore
+-- track_problem_statements.md
```

## Key Findings

1. **Bimodal narrative distribution**: 36% of reports are Type-A (minimal, 32 chars), 64% are Type-B (detailed, ~1,200 chars). Only Type-B reports are viable for summarization.
2. **Entity coverage is high**: >95% of Type-B reports have amounts, dates, customer names, and counterparties available from structured fields — making entity-aware summarization practical.
3. **Cross-border transactions**: ~28% of reports involve multiple countries, adding complexity.
4. **LLM API dominates**: The 70B-parameter cloud model achieves perfect entity preservation and 80% length compliance. Smaller local models (3B) are not yet capable of this task's instruction complexity.
5. **Factual faithfulness > fluency**: In the financial crime domain, dropping a counterparty or amount is far worse than awkward phrasing. The evaluation framework appropriately prioritizes entity preservation.
