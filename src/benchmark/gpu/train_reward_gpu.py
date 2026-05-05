from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from trl import RewardConfig, RewardTrainer

from .common import build_lora_config, ensure_dir, load_tokenizer, pick_dtype, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scalar reward model for matched DDO-RM/PPO/GRPO experiments.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--train_split", default="train_prefs")
    parser.add_argument("--eval_split", default=None)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto", "none"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--center_rewards_coefficient", type=float, default=1e-2)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--report_to", default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    tokenizer = load_tokenizer(args.base_model, padding_side="right", trust_remote_code=args.trust_remote_code)
    dtype = pick_dtype(args.dtype)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = load_dataset(args.dataset_name, split=args.train_split)
    if args.max_train_examples is not None:
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(min(args.max_train_examples, len(train_dataset))))
    eval_dataset = None
    if args.eval_split:
        eval_dataset = load_dataset(args.dataset_name, split=args.eval_split)
        if args.max_eval_examples is not None:
            eval_dataset = eval_dataset.shuffle(seed=args.seed).select(range(min(args.max_eval_examples, len(eval_dataset))))

    peft_config = None
    if args.use_lora:
        peft_config = build_lora_config(
            args.base_model,
            task_type="SEQ_CLS",
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )

    training_args = RewardConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps if args.save_steps > 0 else 500000000,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps" if args.save_steps > 0 else "no",
        center_rewards_coefficient=args.center_rewards_coefficient,
        bf16=args.dtype in {"bf16", "bfloat16"},
        fp16=args.dtype in {"fp16", "float16"},
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    save_json(output_dir / "reward_run_config.json", vars(args))


if __name__ == "__main__":
    main()
