#!/usr/bin/env python
"""Supervised fine-tuning of the planner with LoRA (Table 12).

The planner maps a question to a structured failure-aware plan.  Only the
target plan tokens contribute to the loss.

Example::

    python scripts/train_planner.py --config configs/planner_sft.yaml
    # or override on the command line
    python scripts/train_planner.py --train_file data/planner/train.jsonl \\
        --val_file data/planner/val.jsonl --output_dir checkpoints/flap-planner-7b
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flap.data.datasets import read_jsonl  # noqa: E402
from flap.data.sft import PlannerSFTDataset, collate_fn  # noqa: E402

DEFAULTS = dict(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    train_file="data/planner/train.jsonl",
    val_file="data/planner/val.jsonl",
    output_dir="checkpoints/flap-planner-7b",
    lora_rank=64,
    lora_alpha=128,
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,  # effective batch size 64 on one GPU
    max_seq_length=4096,
    warmup_ratio=0.03,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    gradient_checkpointing=True,
    seed=42,
    merge_adapter=False,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="YAML file with any of the DEFAULTS keys")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=None)
        elif isinstance(v, list):
            parser.add_argument(f"--{k}", nargs="+", default=None)
        else:
            parser.add_argument(f"--{k}", type=type(v), default=None)
    args = parser.parse_args()
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f) or {})
    for k in DEFAULTS:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    return cfg


def main():
    cfg = parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_records = list(read_jsonl(cfg["train_file"]))
    val_records = list(read_jsonl(cfg["val_file"])) if cfg["val_file"] and os.path.exists(cfg["val_file"]) else []
    print(f"train={len(train_records)}  val={len(val_records)}")
    train_ds = PlannerSFTDataset(train_records, tokenizer, cfg["max_seq_length"])
    val_ds = PlannerSFTDataset(val_records, tokenizer, cfg["max_seq_length"]) if val_records else None

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.bfloat16 if cfg["bf16"] else torch.float32,
        trust_remote_code=True,
    )
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        bf16=cfg["bf16"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        eval_strategy=cfg["eval_strategy"] if val_ds is not None else "no",
        optim="adamw_torch",
        seed=cfg["seed"],
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn(tokenizer),
    )
    trainer.train()
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    if cfg["merge_adapter"]:
        merged = model.merge_and_unload()
        merged_dir = os.path.join(cfg["output_dir"], "merged")
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to {merged_dir}")
    print(f"Adapter saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
