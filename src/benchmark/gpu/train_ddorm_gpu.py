from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, get_scheduler

from .common import (
    append_jsonl,
    build_lora_config,
    count_trainable_parameters,
    ensure_dir,
    freeze_model,
    load_tokenizer,
    pick_dtype,
    save_json,
    set_seed,
)
from .data import CandidateCollator, load_candidate_dataset
from .ddorm_math import ddorm_cross_entropy_loss, ddorm_target_distribution
from .scoring import flatten_prompt_candidates, reward_model_scores, sequence_logprobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU DDO-RM finite-candidate distillation training.")

    # Model and data.
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--reward_model_name_or_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_kind", choices=["uf_binarized", "uf_listwise", "nectar"], default="uf_binarized")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_path", default=None, help="Optional load_from_disk path; overrides dataset_name.")
    parser.add_argument("--split", default="train_prefs")
    parser.add_argument("--score_mode", default="mean_rating")
    parser.add_argument("--max_candidates", type=int, default=None)
    parser.add_argument(
        "--candidate_strategy",
        choices=["first", "top_gold", "random", "random_include_best"],
        default="first",
    )
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # DDO-RM objective.
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--decision_temperature", type=float, default=1.0)
    parser.add_argument("--disable_reward_centering", action="store_true")
    parser.add_argument("--distill_steps_per_batch", type=int, default=1)
    parser.add_argument("--use_gold_rewards", action="store_true", help="Use dataset reward_scores instead of a learned RM.")
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--reward_shift", type=float, default=0.0)
    parser.add_argument("--reward_noise_std", type=float, default=0.0)
    parser.add_argument("--reward_standardize_per_prompt", action="store_true")

    # Training.
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--reward_batch_size", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=0)

    # Memory / PEFT.
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto", "none"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def maybe_standardize_rewards(rewards: torch.Tensor) -> torch.Tensor:
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (rewards - mean) / std


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    metrics_path = output_dir / "train_metrics.jsonl"
    accelerator = Accelerator()

    tokenizer = load_tokenizer(args.model_name_or_path, padding_side="right", trust_remote_code=args.trust_remote_code)
    dtype = pick_dtype(args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.use_lora:
        from peft import get_peft_model

        peft_config = build_lora_config(
            args.model_name_or_path,
            task_type="CAUSAL_LM",
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, peft_config)

    reward_model = None
    reward_tokenizer = None
    if not args.use_gold_rewards:
        if not args.reward_model_name_or_path:
            raise ValueError("--reward_model_name_or_path is required unless --use_gold_rewards is set.")
        reward_tokenizer = load_tokenizer(
            args.reward_model_name_or_path,
            padding_side="right",
            trust_remote_code=args.trust_remote_code,
        )
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model_name_or_path,
            num_labels=1,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
        freeze_model(reward_model)

    dataset = load_candidate_dataset(
        dataset_kind=args.dataset_kind,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        split=args.split,
        score_mode=args.score_mode,
        max_candidates=args.max_candidates,
        candidate_strategy=args.candidate_strategy,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=CandidateCollator(),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(dataloader) * args.distill_steps_per_batch / args.gradient_accumulation_steps)
    max_updates = args.max_steps or max(1, int(math.ceil(args.num_train_epochs * updates_per_epoch)))
    total_micro_steps = max_updates * args.gradient_accumulation_steps
    warmup_steps = int(args.warmup_ratio * max_updates)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_updates,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    if reward_model is not None:
        reward_model = reward_model.to(accelerator.device)

    if accelerator.is_main_process:
        save_json(
            output_dir / "run_config.json",
            {
                **vars(args),
                "num_train_examples": len(dataset),
                "num_update_steps": max_updates,
                "trainable_parameters": count_trainable_parameters(model),
            },
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    micro_step = 0
    start_time = time.time()
    progress = tqdm(total=max_updates, disable=not accelerator.is_main_process, desc="DDO-RM updates")

    while global_update < max_updates:
        for batch in dataloader:
            if global_update >= max_updates:
                break
            prompts = batch["prompts"]
            candidates = batch["candidates"]
            batch_size = len(prompts)
            num_candidates = len(candidates[0])
            flat_prompts, flat_candidates = flatten_prompt_candidates(prompts, candidates)

            # Reward scores are fixed for the repeated distillation steps on the same finite set.
            with torch.no_grad():
                if args.use_gold_rewards:
                    rewards = torch.tensor(
                        batch["reward_scores"],
                        device=accelerator.device,
                        dtype=torch.float32,
                    )
                else:
                    assert reward_model is not None and reward_tokenizer is not None
                    rewards = reward_model_scores(
                        reward_model,
                        reward_tokenizer,
                        flat_prompts,
                        flat_candidates,
                        max_length=args.max_length,
                        batch_size=args.reward_batch_size,
                    ).view(batch_size, num_candidates).float()
                rewards = rewards * args.reward_scale + args.reward_shift
                if args.reward_noise_std > 0:
                    rewards = rewards + torch.randn_like(rewards) * args.reward_noise_std
                if args.reward_standardize_per_prompt:
                    rewards = maybe_standardize_rewards(rewards)

            target_q = None
            last_stats = None
            last_loss = None
            last_policy_scores = None

            for _ in range(args.distill_steps_per_batch):
                policy_scores = sequence_logprobs(
                    model,
                    tokenizer,
                    flat_prompts,
                    flat_candidates,
                    max_length=args.max_length,
                    add_eos=True,
                    normalize_by_length=True,
                ).avg_logprob.view(batch_size, num_candidates)

                if target_q is None:
                    with torch.no_grad():
                        _, q, _, stats = ddorm_target_distribution(
                            policy_scores.detach().float(),
                            rewards.detach().float(),
                            eta=args.eta,
                            temperature=args.decision_temperature,
                            center_rewards=not args.disable_reward_centering,
                        )
                        target_q = q.detach().to(policy_scores.dtype)
                        last_stats = stats

                loss = ddorm_cross_entropy_loss(
                    policy_scores,
                    target_q,
                    temperature=args.decision_temperature,
                )
                loss_for_backward = loss / args.gradient_accumulation_steps
                accelerator.backward(loss_for_backward)
                micro_step += 1
                last_loss = loss.detach()
                last_policy_scores = policy_scores.detach()

                if micro_step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm and args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_update += 1
                    progress.update(1)

                    if accelerator.is_main_process and (
                        global_update % args.logging_steps == 0 or global_update == 1
                    ):
                        elapsed = max(time.time() - start_time, 1e-6)
                        row = {
                            "update": global_update,
                            "micro_step": micro_step,
                            "loss": float(last_loss.float().item()) if last_loss is not None else None,
                            "lr": float(scheduler.get_last_lr()[0]),
                            "k": num_candidates,
                            "eta": args.eta,
                            "decision_temperature": args.decision_temperature,
                            "eta_over_temperature": args.eta / args.decision_temperature,
                            "examples_per_second_per_process": float(global_update * args.per_device_train_batch_size / elapsed),
                        }
                        if last_stats is not None:
                            row.update(
                                {
                                    "mean_kl_q_p": float(last_stats.mean_kl_q_p.item()),
                                    "mean_target_entropy": float(last_stats.mean_target_entropy.item()),
                                    "mean_policy_entropy": float(last_stats.mean_policy_entropy.item()),
                                    "mean_reward_gain_under_rm": float(
                                        last_stats.mean_reward_gain_under_rm.item()
                                    ),
                                    "mean_abs_centered_reward": float(
                                        last_stats.mean_abs_centered_reward.item()
                                    ),
                                }
                            )
                        if last_policy_scores is not None:
                            row["mean_policy_score"] = float(last_policy_scores.float().mean().item())
                        append_jsonl(metrics_path, row)

                    if args.save_steps and global_update % args.save_steps == 0:
                        ckpt = output_dir / f"checkpoint-{global_update}"
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            unwrapped = accelerator.unwrap_model(model)
                            unwrapped.save_pretrained(ckpt, safe_serialization=True)
                            tokenizer.save_pretrained(ckpt)
                    if global_update >= max_updates:
                        break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        save_json(
            output_dir / "final_metrics.json",
            {
                "updates": global_update,
                "micro_steps": micro_step,
                "elapsed_seconds": time.time() - start_time,
                "output_dir": str(output_dir),
            },
        )
    progress.close()


if __name__ == "__main__":
    main()
