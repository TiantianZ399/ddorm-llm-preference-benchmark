from __future__ import annotations

import argparse

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, DataCollatorWithPadding
from trl.experimental.ppo import PPOConfig, PPOTrainer

from .common import build_lora_config, ensure_dir, load_tokenizer, pick_dtype, save_json, set_seed
from .data import load_prompt_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched reward-model PPO-RLHF baseline using TRL experimental PPO.")
    parser.add_argument("--model_name_or_path", required=True, help="SFT/base policy path.")
    parser.add_argument("--reward_model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ref_model_name_or_path", default=None)
    parser.add_argument("--value_model_name_or_path", default=None)

    parser.add_argument("--dataset_kind", choices=["uf_binarized", "uf_listwise", "nectar"], default="uf_binarized")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--split", default="train_prefs")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--total_episodes", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_ppo_epochs", type=int, default=4)
    parser.add_argument("--num_mini_batches", type=int, default=1)
    parser.add_argument("--local_rollout_forward_batch_size", type=int, default=8)
    parser.add_argument("--response_length", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--kl_estimator", choices=["k1", "k3"], default="k1")
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--missing_eos_penalty", type=float, default=1.0)
    parser.add_argument("--stop_token", choices=["eos", "none"], default="eos")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)

    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto", "none"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
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

    tokenizer = load_tokenizer(args.model_name_or_path, padding_side="left", trust_remote_code=args.trust_remote_code)
    dtype = pick_dtype(args.dtype)

    prompt_ds = load_prompt_dataset(
        dataset_kind=args.dataset_kind,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        split=args.split,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    def tokenize(batch):
        return tokenizer(
            batch["prompt"],
            truncation=True,
            max_length=args.max_prompt_length,
            padding=False,
        )

    train_dataset = prompt_ds.map(tokenize, batched=True, remove_columns=prompt_ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    ref_model = None
    if not args.use_lora:
        ref_path = args.ref_model_name_or_path or args.model_name_or_path
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_path,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        ref_model.config.pad_token_id = tokenizer.pad_token_id

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name_or_path,
        num_labels=1,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    reward_model.config.pad_token_id = tokenizer.pad_token_id

    value_path = args.value_model_name_or_path or args.model_name_or_path
    value_model = AutoModelForSequenceClassification.from_pretrained(
        value_path,
        num_labels=1,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    value_model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = None
    if args.use_lora:
        peft_config = build_lora_config(
            args.model_name_or_path,
            task_type="CAUSAL_LM",
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )

    ppo_args = PPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        total_episodes=args.total_episodes,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_ppo_epochs=args.num_ppo_epochs,
        num_mini_batches=args.num_mini_batches,
        local_rollout_forward_batch_size=args.local_rollout_forward_batch_size,
        response_length=args.response_length,
        temperature=args.temperature,
        kl_coef=args.kl_coef,
        kl_estimator=args.kl_estimator,
        cliprange=args.cliprange,
        vf_coef=args.vf_coef,
        gamma=args.gamma,
        lam=args.lam,
        missing_eos_penalty=args.missing_eos_penalty,
        stop_token=None if args.stop_token == "none" else args.stop_token,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps" if args.save_steps > 0 else "no",
        bf16=args.dtype in {"bf16", "bfloat16"},
        fp16=args.dtype in {"fp16", "float16"},
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        remove_unused_columns=False,
    )

    trainer = PPOTrainer(
        args=ppo_args,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    save_json(output_dir / "ppo_run_config.json", vars(args))


if __name__ == "__main__":
    main()
