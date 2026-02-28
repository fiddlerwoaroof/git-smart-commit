#!/usr/bin/env python3
"""
train.py: LoRA finetune Qwen3-Coder on git-smart-commit SFT data via Unsloth.

Run this on a Vast.ai A100/H100 after installing dependencies:
    pip install unsloth trl transformers accelerate bitsandbytes

Usage:
    python train.py [--model MODEL] [--dataset FILE] [--output DIR]
                   [--lora-rank N] [--lora-alpha N] [--epochs N]
                   [--batch-size N] [--grad-accum N] [--lr LR]
                   [--max-seq-length N] [--val-dataset FILE]
                   [--warmup-ratio FLOAT] [--seed N]

Outputs:
    <output>/          LoRA adapter weights (load with PeftModel)
    <output>/runs/     TensorBoard logs
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA finetune Qwen3-Coder on git-smart-commit SFT data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        default="unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit",
        help="Base model (Hugging Face repo or local path). "
        "Default: unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit",
    )
    p.add_argument(
        "--dataset",
        default="train.jsonl",
        help="Training JSONL from prepare-sft-data (default: train.jsonl)",
    )
    p.add_argument(
        "--val-dataset",
        default="val.jsonl",
        help="Validation JSONL (default: val.jsonl)",
    )
    p.add_argument(
        "--output",
        default="lora-adapter",
        help="Directory to save LoRA adapter (default: lora-adapter)",
    )
    p.add_argument(
        "--lora-rank", type=int, default=32, help="LoRA rank r (default: 32)"
    )
    p.add_argument(
        "--lora-alpha", type=int, default=64, help="LoRA alpha (default: 64)"
    )
    p.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    p.add_argument(
        "--batch-size", type=int, default=4, help="Per-device batch size (default: 4)"
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4, effective batch = batch-size × grad-accum)",
    )
    p.add_argument(
        "--lr", type=float, default=1.5e-4, help="Learning rate (default: 1.5e-4)"
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Max token sequence length (default: 8192)",
    )
    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.10,
        help="Fraction of steps used for LR warmup (default: 0.10)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (use full bf16 LoRA instead)",
    )
    return p.parse_args()


def load_jsonl(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"Warning: skipping malformed line {lineno} in {path}: {e}",
                    file=sys.stderr,
                )
    return examples


def main() -> None:
    args = parse_args()

    # Validate inputs
    for path in [args.dataset, args.val_dataset]:
        if not Path(path).exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    # ── Imports (deferred so --help works without GPU) ────────────────────────
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print(
            "Error: unsloth not installed.\n"
            "Install with: pip install unsloth trl transformers accelerate bitsandbytes",
            file=sys.stderr,
        )
        sys.exit(1)

    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
        dtype=None,  # auto-detect
    )

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)",
        file=sys.stderr,
    )

    # ── Load and format datasets ──────────────────────────────────────────────
    def apply_chat_template(examples: dict) -> dict:
        texts = []
        for messages in examples["messages"]:
            # Use the tokenizer's built-in chat template.
            # enable_thinking=False disables Qwen3's chain-of-thought prefix,
            # keeping sequences shorter and the output format predictable.
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            except TypeError:
                # Fallback for tokenizers without enable_thinking
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            texts.append(text)
        return {"text": texts}

    print(f"Loading training data: {args.dataset}", file=sys.stderr)
    train_raw = load_jsonl(args.dataset)
    train_ds = Dataset.from_list(train_raw).map(
        apply_chat_template,
        batched=True,
        remove_columns=["messages"],
    )

    print(f"Loading validation data: {args.val_dataset}", file=sys.stderr)
    val_raw = load_jsonl(args.val_dataset)
    val_ds = Dataset.from_list(val_raw).map(
        apply_chat_template,
        batched=True,
        remove_columns=["messages"],
    )

    print(
        f"Train: {len(train_ds)} examples  Val: {len(val_ds)} examples", file=sys.stderr
    )

    # ── Training ──────────────────────────────────────────────────────────────
    output_dir = args.output
    logging_dir = str(Path(output_dir) / "runs")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=True,  # pack short sequences together for efficiency
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            bf16=True,
            fp16=False,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            seed=args.seed,
            report_to="tensorboard",
            logging_dir=logging_dir,
            dataloader_num_workers=2,
        ),
    )

    print("Starting training…", file=sys.stderr)
    trainer_stats = trainer.train()

    print(
        f"\nTraining complete. "
        f"Runtime: {trainer_stats.metrics['train_runtime']:.0f}s  "
        f"Samples/s: {trainer_stats.metrics['train_samples_per_second']:.1f}",
        file=sys.stderr,
    )

    # ── Save adapter ──────────────────────────────────────────────────────────
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapter saved to: {output_dir}", file=sys.stderr)

    # Save training config alongside adapter for reproducibility
    config_path = Path(output_dir) / "training_config.json"
    config = vars(args)
    config["train_examples"] = len(train_ds)
    config["val_examples"] = len(val_ds)
    config["train_runtime_s"] = trainer_stats.metrics.get("train_runtime")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to: {config_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
