# Track 6 Execution Plan - AI-Powered STR Summarization

## 🎯 Mission

Build an AI system that automatically summarizes long financial crime narratives (1,000-1,300 characters) into concise analyst-facing summaries (100-200 words) while preserving all critical facts.

---

## 📊 Dataset Snapshot

| Metric | Value |
|--------|-------|
| Total Reports | 276 STR XML files |
| Narrative Types | 2 (minimal + detailed) |
| Mean Narrative Length | 807 characters |
| Compression Ratio | ~4-6x (1,200 → 150 words) |
| Currency | All NPR (Nepali Rupee) |
| Time Period | Oct-Nov 2022 |

**Key Challenge:** 36% of reports have zero narrative ("Suspicious transaction observed.") — these require pure structured field summarization.

---

## 🛠️ Implementation Phases

### Phase 1: Data Preparation & Extraction (Days 1-2)

**Task 1.1:** Create data pipeline
```python
# Load all 276 reports
# Extract: report_id, narrative, amount, date, names, transaction_mode
# Split: Type-A (minimal) vs Type-B (detailed) reports
# Output: Pandas DataFrame for analysis
```

**Task 1.2:** Build entity extraction module
```python
# Extract key entities using spaCy:
# - Customer names
# - Bank/Institution names
# - Counterparty names
# - Amounts (with currency)
# - Dates
# - Transaction modes
```

**Task 1.3:** Create evaluation framework
```python
# Metrics to implement:
# 1. ROUGE-L (fluency)
# 2. BERTScore (semantic similarity)
# 3. Entity Preservation Score (critical entities present?)
# 4. Length (must be 100-200 words)
```

---

### Phase 2: Baseline Development (Days 3-5)

**Task 2.1:** Extractive Summarization (Low Risk)
```python
# Approach: Select key sentences from narrative
# Algorithm:
#   1. Tokenize narrative into sentences
#   2. Score each sentence by importance (TF-IDF)
#   3. Include sentences with critical entities
#   4. Select top-K sentences until word limit reached
#
# Expected Performance:
#   - Highly faithful (no hallucinations)
#   - May be less fluent
#   - Good baseline (60-70% ROUGE-L)
```

**Task 2.2:** Hybrid Approach (Medium Risk)
```python
# Extract template from narrative pattern:
# "Customer [NAME] at [BANK] conducted [N] [MODE] transactions 
#  totaling [AMOUNT] on [DATE] to [COUNTERPARTIES]. 
#  Suspicious pattern: [PATTERN]."
#
# Fill slots with extracted entities
# Result: Consistent, accurate 80-100 word summaries
```

---

### Phase 3: Advanced Models (Days 6-10)

**Task 3.1:** BART Fine-tuning
```python
# IF gold summaries available:
#   1. Fine-tune BART on human summaries (80/20 split)
#   2. Control output length with max_tokens
#   3. Evaluate on test set
#   4. Compare to extractive baseline
#
# Expected Performance:
#   - Better fluency (ROUGE-L: 70-80%)
#   - Risk of hallucination (extra facts)
#   - Need validation against structured fields
```

**Task 3.2:** LLM-based Summarization (Alternative)
```python
# Use GPT-3.5 or local LLM with structured prompts:
# "Summarize this STR narrative in 150 words. 
#  MUST include: customer name, amount, date, counterparties, 
#  suspicious pattern. Do NOT invent any facts."
#
# Pros: High quality, minimal training
# Cons: External dependency, cost, latency
```

---

### Phase 4: Validation & Optimization (Days 11-12)

**Task 4.1:** Factual Faithfulness Checks
```python
# 1. Cross-reference summary vs. structured XML fields
# 2. Flag any amount mismatches, date shifts, missing parties
# 3. Penalize hallucinated facts
# 4. Implement confidence scoring
```

**Task 4.2:** Human Review
```python
# Manually review 20-30 summaries from each approach
# Assess:
#   - Does analyst understand the case in 30 seconds?
#   - Are all critical facts preserved?
#   - Is the length appropriate?
```

---

## 📋 Deliverables Checklist

- [ ] **Data Pipeline** (loading, parsing, extraction)
- [ ] **EDA Notebook** (exploratory data analysis + visualizations)
- [ ] **Evaluation Framework** (ROUGE, BERTScore, entity metrics)
- [ ] **Extractive Baseline** (sentence selection + template)
- [ ] **Abstractive Model** (BART or LLM-based)
- [ ] **Validation Script** (factual accuracy checks)
- [ ] **Live Demo** (on 5-10 sample reports)
- [ ] **Technical Documentation** (architecture, design decisions)
- [ ] **Presentation Slides** (problem, approach, results)
- [ ] **GitHub Repository** (clean, documented code)

---

## 🔑 Critical Success Factors

1. **Preserve ALL critical facts** — no dropped names, amounts, or dates
2. **Stay within word limit** — 100-200 words, no exceptions
3. **Analyst-ready** — summary must be actionable in <30 seconds
4. **Scalable** — must process all 276 reports in <1 minute
5. **Explainable** — show which features drove the summary

---

## 📁 Code Structure (to implement)

```
AIMLHakathon/
├── explore_xml.py                  # ✅ Complete
├── TRACK6_ANALYSIS.md              # ✅ Complete
├── TRACK6_EXECUTION_PLAN.md        # ✅ This file
│
├── data/
│   └── reports/                    # ✅ 276 XML files
│
├── src/
│   ├── data_loader.py              # TODO: Load + parse XML
│   ├── entity_extractor.py         # TODO: Extract names, amounts, dates
│   ├── summarizer_extractive.py    # TODO: Sentence selection baseline
│   ├── summarizer_hybrid.py        # TODO: Template-based approach
│   ├── summarizer_abstractive.py   # TODO: BART/LLM wrapper
│   └── evaluator.py                # TODO: ROUGE, BERTScore, entity checks
│
├── notebooks/
│   └── eda_and_analysis.ipynb      # TODO: Full exploratory analysis
│
├── evaluation/
│   ├── gold_summaries.csv          # TODO: Acquire or create
│   └── results.json                # TODO: Evaluation metrics
│
├── demo.py                         # TODO: Live demo script
├── requirements.txt                # TODO: Dependencies
├── README.md                       # TODO: Project overview
└── TECHNICAL_DOCUMENTATION.md      # TODO: Design & methods
```

---

## 📝 Recommended First Step

**Create `data_loader.py`:**

```python
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

def load_all_reports(reports_dir):
    """Load all STR reports into a DataFrame."""
    reports = []
    
    for xml_file in Path(reports_dir).glob("report_*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        report = {
            'report_id': root.findtext('report_id'),
            'narrative': root.findtext('reason', ''),
            'amount': root.findtext('.//transaction/amount_local'),
            'date': root.findtext('.//transaction/date_transaction'),
            'from_account': root.findtext('.//t_from_my_client/from_account/account_name'),
            'to_account': root.findtext('.//t_to/to_account/account_name'),
            'bank': root.findtext('.//from_account/institution_name'),
            'mode': root.findtext('.//transaction/transmode_comment'),
        }
        
        reports.append(report)
    
    return pd.DataFrame(reports)

# Usage:
# df = load_all_reports('reports/')
# df.to_csv('all_reports.csv', index=False)
```

---

## ⏱️ Timeline

| Phase | Days | Status |
|-------|------|--------|
| Data Prep | 1-2 | Ready to start |
| Baseline | 3-5 | After Phase 1 |
| Advanced | 6-10 | After baseline works |
| Validation | 11-12 | Final polish |
| **Total** | **~10-12 days** | **Before submission** |

---

## 🚀 You're Ready!

You have:
- ✅ Clear understanding of data (276 reports, bimodal narratives)
- ✅ Key facts to preserve (names, amounts, dates, parties, patterns)
- ✅ Multiple approaches (extractive → hybrid → abstractive)
- ✅ Evaluation framework (ROUGE + entity preservation)
- ✅ Implementation roadmap (4 phases, clear tasks)

**Next: Start with `data_loader.py` and begin Phase 1!**
