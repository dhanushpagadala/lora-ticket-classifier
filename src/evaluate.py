"""
Evaluates and compares:
  (a) the raw base model, prompted zero-shot to classify tickets
  (b) the same base model + your trained LoRA adapter

on the held-out test split, reporting accuracy, macro-F1, and a confusion
matrix, so you have real before/after numbers.

Usage:
    python src/evaluate.py --config configs/train_config.yaml --adapter outputs/adapter
"""
import argparse
import json
import re

import torch
import yaml
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "You are a support-ticket triage classifier. Read the ticket and respond "
    "with exactly one label from this list, and nothing else: "
    "billing, technical, account, feature_request, general."
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_base_model(config):
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config["use_qlora"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
    return model, tokenizer


def predict_label(model, tokenizer, ticket_text, labels, max_new_tokens=8):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ticket_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().lower()

    # Match generated text to the closest known label (handles cases where
    # the model adds punctuation/extra words instead of the bare label).
    for label in labels:
        if label in generated:
            return label
    return "UNPARSEABLE:" + generated[:30]


def run_eval(model, tokenizer, test_rows, labels, tag):
    preds, gold = [], []
    for row in test_rows:
        pred = predict_label(model, tokenizer, row["text"], labels)
        preds.append(pred)
        gold.append(row["label"])

    acc = accuracy_score(gold, preds)
    print(f"\n===== {tag} =====")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(gold, preds, labels=labels, zero_division=0))
    print("Confusion matrix (rows=gold, cols=pred):")
    print(labels)
    print(confusion_matrix(gold, preds, labels=labels))
    return acc, preds, gold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--adapter", type=str, required=True,
                         help="path to trained LoRA adapter (e.g. outputs/adapter)")
    parser.add_argument("--raw_test_path", type=str, default="data/raw/test.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = config["labels"]
    test_rows = load_jsonl(args.raw_test_path)

    print(f"Evaluating on {len(test_rows)} held-out test examples...")

    # --- Base model (zero-shot) ---
    base_model, tokenizer = build_base_model(config)
    base_acc, _, _ = run_eval(base_model, tokenizer, test_rows, labels, "BASE MODEL (zero-shot)")

    # --- Fine-tuned model (base + LoRA adapter) ---
    finetuned_model = PeftModel.from_pretrained(base_model, args.adapter)
    ft_acc, _, _ = run_eval(finetuned_model, tokenizer, test_rows, labels, "FINE-TUNED MODEL (LoRA)")

    print("\n===== SUMMARY =====")
    print(f"{'':16}{'base_model':>12}{'finetuned':>12}")
    print(f"{'accuracy':16}{base_acc:>12.3f}{ft_acc:>12.3f}")
    print(f"\nImprovement: {(ft_acc - base_acc) * 100:+.1f} percentage points")


if __name__ == "__main__":
    main()
