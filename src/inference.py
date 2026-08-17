"""
Quick single-ticket inference with the fine-tuned model.

Usage:
    python src/inference.py --adapter outputs/adapter --text "My card was charged twice this month."
"""
import argparse

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from evaluate import SYSTEM_PROMPT, predict_label  # reuse the same logic used in eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    labels = config["labels"]

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config["use_qlora"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model_name"], quantization_config=bnb_config, device_map="auto"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model_name"], torch_dtype=torch.bfloat16, device_map="auto"
        )

    model = PeftModel.from_pretrained(base_model, args.adapter)

    pred = predict_label(model, tokenizer, args.text, labels)
    print(f"\nTicket: {args.text}")
    print(f"Predicted label: {pred}")


if __name__ == "__main__":
    main()
