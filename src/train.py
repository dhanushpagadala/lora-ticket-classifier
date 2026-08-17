"""
LoRA / QLoRA fine-tuning of a small open-source instruction model on the
ticket-classification dataset produced by data/generate_dataset.py +
src/format_data.py.

Usage:
    python src/train.py --config configs/train_config.yaml
"""
import argparse

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(config):
    model_name = config["model_name"]
    use_qlora = config["use_qlora"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_qlora:
        # 4-bit NormalFloat quantization of the base model. This is what lets
        # a 7B model fit on a 16GB (or even free-tier 15GB T4) GPU: the frozen
        # base weights live in 4-bit, only the small LoRA adapters train in
        # higher precision.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.bfloat16 if config["training"]["bf16"] else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

    model.config.use_cache = False  # required for gradient checkpointing compatibility

    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # sanity check: should be ~0.1-1% of total params

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    t_cfg = config["training"]
    d_cfg = config["data"]

    model, tokenizer = build_model_and_tokenizer(config)

    train_ds = load_dataset("json", data_files=d_cfg["train_path"], split="train")
    val_ds = load_dataset("json", data_files=d_cfg["val_path"], split="train")

    sft_config = SFTConfig(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        weight_decay=t_cfg["weight_decay"],
        logging_steps=t_cfg["logging_steps"],
        eval_strategy=t_cfg["eval_strategy"],
        save_strategy=t_cfg["save_strategy"],
        save_total_limit=t_cfg["save_total_limit"],
        bf16=t_cfg["bf16"],
        fp16=t_cfg["fp16"],
        max_seq_length=t_cfg["max_seq_length"],
        packing=t_cfg["packing"],
        seed=t_cfg["seed"],
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(t_cfg["output_dir"])
    tokenizer.save_pretrained(t_cfg["output_dir"])
    print(f"\nAdapter + tokenizer saved to {t_cfg['output_dir']}")


if __name__ == "__main__":
    main()
