"""
Converts data/raw/{train,val,test}.jsonl (schema: {"text":..., "label":...})
into instruction-tuning format suitable for TRL's SFTTrainer, using the
chat template of whatever model you're fine-tuning.

Each example becomes a chat conversation:
    system: instructions + label set
    user:   the raw ticket
    assistant: the correct label (this is what the model learns to produce)

Usage:
    python src/format_data.py --config configs/train_config.yaml
"""
import argparse
import json
from pathlib import Path

import yaml
from transformers import AutoTokenizer

SYSTEM_PROMPT = (
    "You are a support-ticket triage classifier. Read the ticket and respond "
    "with exactly one label from this list, and nothing else: "
    "billing, technical, account, feature_request, general."
)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_chat_example(row, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["text"]},
        {"role": "assistant", "content": row["label"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text, "label": row["label"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/formatted")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        raw_path = raw_dir / f"{split_name}.jsonl"
        rows = load_jsonl(raw_path)
        formatted = [to_chat_example(r, tokenizer) for r in rows]
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for r in formatted:
                f.write(json.dumps(r) + "\n")
        print(f"Formatted {len(formatted)} examples -> {out_path}")


if __name__ == "__main__":
    main()
