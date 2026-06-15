# Track 6: AI-Powered STR Summarization - Dataset Analysis

**Date:** June 15, 2026  
**Dataset Explored:** 276 Suspicious Transaction Reports (STRs) in XML format

---

## Executive Summary

The dataset contains **276 STR reports** with significant variance in narrative quality and length. The task is to build an AI system that summarizes long reporting-officer narratives (1,000-1,291 characters) into concise 100-200 word summaries while preserving all critical facts.

---

## Key Findings

### 1. Dataset Overview
- **Total Reports:** 276 STR files
- **File Format:** XML (report_000001.xml through report_000276.xml)
- **Geographic Focus:** Nepal (NPR currency)
- **Transaction Types:** Cross-border, Cheque, ACH, Cash Deposit, Card transfers

### 2. Narrative Structure & Quality

#### Two Distinct Narrative Types

**Type A: Minimal Narratives (36.2% of reports)**
- Count: 100 reports
- Narrative Length: 32 characters (e.g., "Suspicious transaction observed.")
- Challenge: These contain no detail and require context from structured fields only
- Example: RPT-2026-000001

**Type B: Detailed Narratives (63.8% of reports)**
- Count: 176 reports  
- Narrative Length: 1,000-1,291 characters
- Challenge: Condense while preserving key facts
- Example: RPT-2026-000185

#### Narrative Statistics
```
Mean:       807.3 chars
Median:   1,236.0 chars  
Min:          32 chars
Max:       1,291 chars
Std Dev:    585.6 chars
```

### 3. Narrative Content Patterns

Detailed narratives follow a consistent template:

```
"During the branch's periodic transaction-monitoring review, the compliance desk at [BANK] 
examined the account held by [CUSTOMER NAME] after an internal threshold alert was triggered. 
Between [DATE] and [DATE], the customer conducted [N] transaction(s) — predominantly [MODE] — 
amounting to approximately [AMOUNT]. [Geographic details]. The principal counterparties 
observed were [NAMES]. For context, the customer's KYC file lists a [profile]. In the reviewing 
officer's judgement [assessment]. It is worth noting that [risk factors]. Nonetheless, 
[suspicious pattern description and conclusion]."
```

### 4. Key Facts to Extract & Preserve

Every summary must include:

| Fact Type | Example | Status |
|-----------|---------|--------|
| **Customer Name** | Stephanie Figueroa | Critical |
| **Bank/Institution** | HBL, KUMARI, BOKL | Critical |
| **Transaction Amount** | NPR 7,710,481 | Critical |
| **Date(s)** | 2022-11-03 | Critical |
| **Transaction Mode** | ACH transfer, Cross-border | Critical |
| **Counterparties** | Tiffany Castillo, Nicholas Brown | Critical |
| **Suspicious Pattern** | Structuring, rapid succession, below-threshold | Critical |
| **Risk Assessment** | Inconsistent with KYC profile | Important |

### 5. Structured XML Fields Available

```xml
<report>
  <report_id>                  # Unique identifier
  <entity_reference>           # Bank reference
  <submission_date>            # Filing date
  <reason>                     # Free-text narrative
  <currency_code_local>        # NPR
  <transaction>
    <transactionnumber>        # Transaction ID
    <amount_local>             # Amount in local currency
    <date_transaction>         # Transaction date/time
    <transmode_comment>        # Type of transfer
    <t_from_my_client>         # Source account details
      <account_name>           # Sending account holder
      <account>                # Account number
      <institution_name>       # Bank name
    </t_from_my_client>
    <t_to>                     # Destination account details
      <account_name>           # Receiving account holder
      <account>                # Account number
      <institution_name>       # Bank name
    </t_to>
  </transaction>
</report>
```

### 6. Transaction Mode Distribution

| Mode | Count |
|------|-------|
| Cross-border transfer | 66 |
| Cheque transfer | 64 |
| ACH transfer | 51 |
| Cash Deposit transfer | 47 |
| Cash Withdrawal transfer | 18 |
| Debit card transfer | 15 |
| Credit card transfer | 15 |
| Other | 0 |

---

## Challenges & Opportunities

### Challenges

1. **Bimodal Distribution:** 100 reports have no useful narrative (requires different handling)
2. **Information Density:** 1,200+ chars → 150 words (75% reduction)
3. **Factual Accuracy:** No invented amounts, parties, or dates
4. **Entity Recognition:** Need to extract/link bank names, account holders, counterparties
5. **Pattern Recognition:** Identify suspicious patterns (structuring, timing, amounts)

### Opportunities

1. **Templated Content:** Narratives follow predictable structure → extractive approach may work
2. **Structured Context:** XML fields provide ground truth for key facts
3. **Consistent Facts:** Same data repeated in both narrative and structured fields → validation possible
4. **Clear Audience:** Analyst-facing summaries have specific information needs

---

## Approach Strategy

### Phase 1: Exploratory (Weeks 1-2)
- [ ] Load gold reference summaries (if available)
- [ ] Analyze quality metrics: ROUGE-L, BERTScore, entity preservation
- [ ] Identify what makes a "good" summary

### Phase 2: Development (Weeks 2-4)
- [ ] Build extractive baseline (select key sentences from narrative)
- [ ] Engineer entity extraction (names, amounts, dates)
- [ ] Implement factual faithfulness checker
- [ ] Build abstractive model (BART/T5 fine-tuning or LLM prompting)

### Phase 3: Refinement (Weeks 4-5)
- [ ] Optimize for 100-200 word constraint
- [ ] Validate against gold summaries
- [ ] Minimize hallucinations
- [ ] Ensure all critical facts preserved

### Phase 4: Evaluation & Demo (Week 5)
- [ ] Create EDA notebook
- [ ] Write technical documentation
- [ ] Live demo on sample reports
- [ ] Prepare presentation

---

## Technical Implementation Roadmap

### Tools & Libraries
```python
# XML Parsing
xml.etree.ElementTree

# NLP & Summarization
transformers (HuggingFace) → BART, T5, GPT
rouge_score                 → Evaluation metrics
bert_score                  → Semantic similarity

# Entity Extraction
spacy                       → NER for persons, orgs, dates, money
regex                       → Pattern-based extraction

# Evaluation
nltk                        → Text metrics
numpy, pandas               → Data analysis
```

### Recommended Models

| Model | Approach | Pros | Cons |
|-------|----------|------|------|
| **Extractive** (baseline) | Select sentences from narrative | Fast, factually safe, interpretable | May not reduce verbosity enough |
| **BART (fine-tuned)** | Abstractive transformer on similar data | Good quality, controllable length | Needs labeled examples, hallucination risk |
| **GPT-based (prompt)** | Structured prompts to GPT-3.5/4 | High quality, minimal training | Cost per summary, slower, external dependency |
| **Hybrid** | Extract entities + templated summary | Best factual accuracy | Requires template engineering |

---

## Next Steps

1. **Get gold summaries** → Request from organizers or sample 20-30 for manual annotation
2. **Set up evaluation framework** → ROUGE + BERTScore + entity preservation metrics
3. **Build extractive baseline** → Establish performance floor
4. **Explore abstractive options** → BART fine-tuning vs. LLM prompting
5. **Iterate on quality** → Balance fluency, conciseness, and faithfulness

---

## File Structure

```
AIMLHakathon/
├── track_problem_statements.md
├── explore_xml.py
├── reports/
│   ├── report_000001.xml
│   ├── report_000002.xml
│   └── ... (276 total)
├── data_analysis_notebook.ipynb    # To create
├── summarization_model.py          # To create
├── evaluation_metrics.py            # To create
└── README.md                        # To create
```

---

**Status:** Ready to proceed with gold summary acquisition and baseline development.
