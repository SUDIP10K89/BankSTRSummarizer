# AML Report Summarization — BART Fine-Tuning

## Why BART, not BERT?

| Model | Architecture | Can generate text? | Use case |
|-------|-------------|-------------------|----------|
| BERT  | Encoder-only | ❌ No | Classification, NER, QA |
| BART  | Encoder-Decoder | ✅ Yes | Summarization, Translation |
| T5    | Encoder-Decoder | ✅ Yes | Summarization, Translation |

**BERT cannot generate summaries** — it produces only hidden states.
BART (`facebook/bart-base`) is the standard choice for summarization fine-tuning.

---

## Dataset Schema

Your CSV has two report types:

| Type | Narrative | Strategy |
|------|-----------|----------|
| Type-A | `"Suspicious transaction observed."` (32 chars) | Structured fields → summary |
| Type-B | Long compliance narrative (~1200 chars) | Compress to 1–2 sentences |

---

## Setup

```bash
pip install -r requirements.txt
python finetune_bart_aml.py
```

GPU recommended. On CPU, each epoch takes ~30–60s for 10 samples.

---

## What the script does

### Step 1 — Label generation (`build_summary`)
Since the dataset has no human-written summaries, we auto-generate
structured reference summaries from the structured columns:

```
Suspicious Cash Withdrawal transfer of NPR 21,807.13 from Sally Dominguez 
at ADBL to Henry King at HBL, executed in Birgunj (cross-border: GB→IN). 
Flagged for possible money-laundering indicators including threshold 
avoidance and rapid succession of transfers.
```

### Step 2 — Tokenization
- Source (`narrative`) → max 512 tokens
- Target (summary) → max 128 tokens
- Padding token ids replaced with -100 so loss ignores them

### Step 3 — Training
- Model: `facebook/bart-base`
- Epochs: 10 (small dataset needs more epochs)
- Batch size: 2
- Metric: ROUGE-L (token-overlap F1)
- Early stopping patience: 3 epochs

### Step 4 — Inference
Beam search (num_beams=4) for best quality output.

---

## Scaling up

With only 10 samples, the model will overfit and produce near-memorized
outputs. To train properly:

1. **Generate 500–5000 rows** from your full XML reports
2. **80/10/10 train/val/test split**
3. Use `rouge_score` library for proper ROUGE-1/2/L metrics
4. Consider `facebook/bart-large-cnn` (pre-trained on CNN/DailyMail)
   as base — it already knows summarization, needs less data to adapt

```python
# Switch to the stronger base for better zero-shot quality:
MODEL_NAME = "facebook/bart-large-cnn"
```

---

## Output

Saved to `./aml_bart_finetuned/`:
- `config.json`
- `pytorch_model.bin`
- `tokenizer_config.json` + `vocab.json`

Load later:
```python
from transformers import BartForConditionalGeneration, BartTokenizer
model = BartForConditionalGeneration.from_pretrained("./aml_bart_finetuned")
tokenizer = BartTokenizer.from_pretrained("./aml_bart_finetuned")
```
