# lora-ticket-classifier

Fine-tunes a small open-source LLM (Llama-3.2-1B/3B-Instruct or Phi-3-mini) with
**LoRA / QLoRA** (via Hugging Face `transformers` + `peft` + `trl`) to classify
customer support tickets into 5 categories, and evaluates it head-to-head
against the untuned base model (zero-shot).

This is a genuinely trainable pipeline, not a toy — it does real gradient
updates on real adapter weights. You need a GPU to run it (a free Colab T4 is
enough for the 1B/3B models with QLoRA).

## Task

**Input:** a raw customer support ticket (subject + body)
**Output:** one label from `{billing, technical, account, feature_request, general}`

This mirrors a real production task (ticket routing / triage) and is the kind
of thing companies fine-tune small models for instead of paying per-token API
costs on every incoming ticket.

## Why this is a stronger resume project than "I called the OpenAI API"

- You write and curate the labeled dataset yourself (data quality is 80% of
  the job in real ML work).
- You run an actual training loop — loss curves, hyperparameters, LoRA rank/alpha,
  gradient accumulation — not just prompt engineering.
- You do quantitative before/after evaluation (accuracy, F1 per class,
  confusion matrix) instead of eyeballing outputs.
- You can talk in an interview about *why* LoRA (parameter-efficient, ~0.1-1%
  of weights trained, fits on consumer/free-tier GPUs) vs full fine-tuning.

## Project layout

All files live in one flat folder — no subdirectories:

```
lora-ticket-classifier/
├── README.md
├── requirements.txt
├── train_config.yaml       # all hyperparameters in one place
├── generate_dataset.py     # builds raw_train/raw_val/raw_test.jsonl
├── format_data.py          # turns raw tickets into instruction-tuning format
├── train.py                 # LoRA/QLoRA fine-tuning (PEFT + TRL SFTTrainer)
├── evaluate.py               # base model vs fine-tuned model comparison
├── inference.py              # quick single-ticket prediction CLI
└── colab_quickstart.ipynb    # run the whole pipeline on a free Colab GPU
```

Running the pipeline will generate these alongside the code (not committed —
add to `.gitignore`):

```
raw_train.jsonl / raw_val.jsonl / raw_test.jsonl       # from generate_dataset.py
formatted_train.jsonl / formatted_val.jsonl / formatted_test.jsonl   # from format_data.py
adapter/                                                # trained LoRA weights, from train.py
```

## Quickstart (Colab — recommended if you don't have a local GPU)

1. Upload all the files in this repo to your Colab session (or `git clone` the repo).
2. Runtime → Change runtime type → **T4 GPU**.
3. Open `colab_quickstart.ipynb` and run all cells. It will:
   - install deps
   - generate the labeled dataset
   - fine-tune with QLoRA (4-bit)
   - evaluate base vs fine-tuned
   - save the adapter to `adapter/`

## Quickstart (local GPU machine, e.g. RTX 3060+/cloud instance)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Generate the labeled dataset (600 examples, deterministic w/ seed)
python generate_dataset.py --n_per_class 120 --seed 42

# 2. Convert to instruction-tuning JSONL format
python format_data.py --config train_config.yaml

# 3. Fine-tune with LoRA (or QLoRA if VRAM-constrained — see config)
python train.py --config train_config.yaml

# 4. Evaluate base model vs your fine-tuned model
python evaluate.py --config train_config.yaml --adapter adapter

# 5. Try it on a single ticket
python inference.py --adapter adapter --text "My card was charged twice this month, please refund."
```

## Swapping in your OWN real data instead of synthetic data

The synthetic generator in `generate_dataset.py` exists so the pipeline runs
end-to-end out of the box. For a real resume project, **replace it with your
own 500-1000 labeled examples** — e.g. pull anonymized tickets from a public
dataset (Kaggle has several customer-support-ticket datasets) or hand label
real examples. Keep the same JSONL schema:

```json
{"text": "Subject: ...\nBody: ...", "label": "billing"}
```

Save it as `raw_train.jsonl` / `raw_val.jsonl` / `raw_test.jsonl` and
everything downstream (`format_data.py`, `train.py`, `evaluate.py`) works
unchanged. This swap is what turns it from "ran a tutorial" into "curated my
own dataset" for your resume bullet.

## Model choice

Default config uses `meta-llama/Llama-3.2-1B-Instruct` (fast to train, fits
free Colab). Swap `model_name` in `train_config.yaml` to:
- `microsoft/Phi-3-mini-4k-instruct` (3.8B, stronger baseline, still QLoRA-able on T4)
- `mistralai/Mistral-7B-Instruct-v0.3` (needs QLoRA 4-bit + a T4/A10/A100)

Llama models require accepting the license on Hugging Face and running
`huggingface-cli login` first (or `notebook_login()` in Colab).

## What "success" looks like

`evaluate.py` prints a side-by-side table like:

```
                base_model   finetuned
accuracy        0.42         0.91
macro_f1        0.38         0.90
```

Save this table (and a confusion matrix) — it's the evidence for your resume
bullet: *"...improving task accuracy by 49 points over the zero-shot base
model."*

## Resume bullet template

> Fine-tuned a 1B–7B open-source LLM (Llama 3.2 / Mistral) using LoRA/QLoRA
> (PEFT) on a self-curated, labeled dataset of N support tickets for
> 5-way intent classification, improving accuracy from X% (zero-shot base
> model) to Y% (+Z pts) and validated with held-out accuracy/F1 evaluation.
