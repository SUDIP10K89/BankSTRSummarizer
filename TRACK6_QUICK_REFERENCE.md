# TRACK 6 - QUICK REFERENCE GUIDE

## 🎯 THE PROBLEM
**Summarize 276 STR financial crime narratives** (1,000-1,300 chars each) → **100-200 word analyst summaries** while preserving ALL critical facts.

---

## 📊 THE DATA

```
276 STR Reports (XML files)
├── Type A: Minimal narratives (36%) - 32 chars ["Suspicious transaction observed."]
└── Type B: Detailed narratives (64%) - 1,000-1,291 chars
    Mean: 807 chars  |  Need to compress to ~150 words
```

---

## 🔑 FACTS YOU MUST PRESERVE

| Must Have | Example |
|-----------|---------|
| **Customer Name** | Stephanie Figueroa |
| **Bank** | HBL, KUMARI |
| **Amount** | NPR 7,710,481 |
| **Date** | 2022-11-03 |
| **Mode** | ACH transfer |
| **Counterparties** | Tiffany Castillo, Nicholas Brown |
| **Suspicious Pattern** | Structuring, rapid succession |

---

## 🛠️ THREE APPROACHES

### 1. **Extractive** (Easy, Safe) ⭐
- Pick key sentences from narrative
- PLUS structured facts from XML
- **Result:** 150-word summary
- **Risk:** Low (can't hallucinate)
- **Quality:** Medium (6-70% ROUGE-L)

### 2. **Hybrid** (Medium, Reliable)
- Extract entities: names, amounts, dates
- Fill template: "Customer [X] at [BANK] sent [AMT] to [Y] on [DATE]. Pattern: [Z]"
- **Result:** Consistent, accurate summaries
- **Risk:** Low
- **Quality:** Medium-High

### 3. **Abstractive** (Hard, Best Quality)
- Fine-tune BART on gold summaries (if available)
- OR use GPT with structured prompts
- **Result:** Natural, fluent summaries
- **Risk:** High (hallucinations possible)
- **Quality:** High (75-85% ROUGE-L)

---

## 📅 BUILD PLAN (12 Days)

```
Week 1:
  Day 1-2: Load data, extract entities, build evaluation metrics
  Day 3-5: Build extractive baseline (GET THIS WORKING FIRST)
  
Week 2:
  Day 6-10: Hybrid OR abstractive approach (try both)
  Day 11-12: Validation, human review, final demo
```

---

## ✅ IMMEDIATE NEXT STEPS

1. Create `data_loader.py` — Load 276 XML reports into DataFrame
2. Create `entity_extractor.py` — Pull out names, amounts, dates
3. Create `evaluator.py` — ROUGE + entity preservation metrics
4. Create `summarizer_extractive.py` — Baseline (sentences + template)
5. Test on 10 reports, evaluate quality

---

## 📁 KEY FILES CREATED

- `explore_xml.py` → Explored dataset structure ✅
- `TRACK6_ANALYSIS.md` → Dataset analysis & findings ✅
- `TRACK6_EXECUTION_PLAN.md` → Detailed roadmap ✅
- `TRACK6_QUICK_REFERENCE.md` → This file ✅

---

## 🎯 EVALUATION METRICS

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **ROUGE-L** | Summary fluency vs. reference | > 0.70 |
| **BERTScore** | Semantic similarity | > 0.80 |
| **Entity Preservation** | All critical facts present? | 100% |
| **Word Count** | Length check | 100-200 words |
| **Hallucination Rate** | False/invented facts | 0% |

---

## 💡 TIPS FOR SUCCESS

✅ **DO:**
- Start with extractive baseline (quick win)
- Validate every summary against XML fields
- Keep structuring simple (no unnecessary ML complexity)
- Test on diverse report types (minimal vs. detailed)

❌ **DON'T:**
- Skip evaluation — you MUST measure quality
- Over-engineer early — baseline first!
- Trust neural models blindly — verify facts
- Ignore the 36% minimal narratives (special handling needed)

---

## 🚀 YOU'RE READY TO CODE!

You understand:
- ✅ What data you have (276 reports, 2 types)
- ✅ What you must preserve (7 critical fact types)
- ✅ How to evaluate (ROUGE + entity metrics)
- ✅ Three viable approaches (extractive → hybrid → abstractive)

**Start coding Phase 1 → Build Phase 1 → Evaluate → Iterate!**
