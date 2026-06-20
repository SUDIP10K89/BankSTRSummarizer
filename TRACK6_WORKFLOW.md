# Track 6 Pipeline Workflow

```mermaid
flowchart TD
    A[STR XML reports<br/>reports/report_*.xml] --> B[data_pipeline.py<br/>parse XML fields]
    B --> C[data/track6_reports_dataset.csv<br/>base report table]
    C --> D[entity_extraction.py<br/>extract critical facts]
    D --> E[data/track6_reports_with_entities.csv<br/>facts + narrative + report type]

    E --> F1[llm_summarizer.py<br/>Llama 3.1 summaries]
    E --> F2[Extractive summarizer<br/>sentence-based baseline]
    E --> F3[Hybrid summarizer<br/>template + facts]

    F1 --> G1[data/track6_llm_summaries.csv]
    F2 --> G2[data/track6_summaries_extractive.csv]
    F3 --> G3[data/track6_summaries_hybrid.csv]

    G1 --> H[evaluation.py<br/>length + fact preservation]
    G2 --> H
    G3 --> H
    E --> H

    H --> I[data/track6_evaluation_results.csv<br/>row-level pass/fail]
    H --> J[data/track6_evaluation_summary.json<br/>aggregate metrics]

    I --> K[demo_output.py<br/>presentation sample]
    K --> L[data/track6_demo_sample.csv]
    K --> M[data/track6_demo_sample.md]
```

## Stage Summary

| Stage | Script | Input | Output | Purpose |
|---|---|---|---|---|
| 1. Data parsing | `data_pipeline.py` | `reports/report_*.xml` | `data/track6_reports_dataset.csv` | Convert XML STR reports into a flat table. |
| 2. Entity extraction | `entity_extraction.py` | `track6_reports_dataset.csv` | `track6_reports_with_entities.csv` | Extract customer, counterparty, banks, amount, date, mode, countries, account numbers. |
| 3. Summarization | `llm_summarizer.py` | `track6_reports_with_entities.csv` | `track6_llm_summaries.csv` | Generate 100-200 word analyst-facing summaries. |
| 4. Evaluation | `evaluation.py` | summaries + entity dataset | `track6_evaluation_results.csv`, `track6_evaluation_summary.json` | Check word count and factual faithfulness. |
| 5. Demo output | `demo_output.py` | `track6_evaluation_results.csv` | `track6_demo_sample.csv`, `track6_demo_sample.md` | Build final presentation-ready examples. |

## Main Commands

```bash
python data_pipeline.py
python entity_extraction.py
python llm_summarizer.py --limit 5 --overwrite
python evaluation.py
python demo_output.py --sample-size 10
```

For larger runs:

```bash
python llm_summarizer.py --all --resume --batch-size 25
python evaluation.py
python demo_output.py --sample-size 10
```

## Validation Logic

The evaluation stage checks:

- summary length is 100-200 words
- customer is preserved
- counterparty is preserved
- bank/institution names are preserved
- amount is preserved, allowing comma formatting differences
- date is preserved
- transaction mode is preserved
- account numbers are preserved

The final demo output shows:

- original narrative
- extracted facts
- generated summary
- validation result
- pass/fail explanation
