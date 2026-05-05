from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification
from trl import GRPOConfig, GRPOTrainer

from .common import build_lora_config, ensure_dir, freeze_model, load_tokenizer, pick_dtype, save_json, set_seed
from .data import load_prompt_dataset, render_messages
from .scoring import reward_model_scores


def _obj_to_text(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        if all(isinstance(x, dict) for x in obj):
            return render_messages(obj, add_assistant_prefix=False)
        return "".join(_obj_to_text(x) for x in obj)
    if isinstance(obj, dict):
        return str(obj.get("content", obj))
    return str(obj)


class RewardModelScorer:
    """Callable reward function for TRL GRPOTrainer using a scalar reward model."""

    def __init__(
        self,
        reward_model_name_or_path: str,
        *,
        dtype: str = "bf16",
        max_length: int = 1024,
        batch_size: int = 8,
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
        trust_remote_code: bool = False,
    ) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer(
            reward_model_name_or_path,
            padding_side="right",
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name_or_path,
            num_labels=1,
            torch_dtype=pick_dtype(dtype),
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        freeze_model(self.model)
        self.max_length = max_length
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.reward_shift = reward_shift

    def __call__(self, prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float | None]:
        prompt_texts = [_obj_to_text(x) for x in prompts]
        completion_texts = [_obj_to_text(x) for x in completions]
        scores = reward_model_scores(
            self.model,
            self.tokenizer,
            prompt_texts,
            completion_texts,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
        scores = scores.float() * self.reward_scale + self.reward_shift
        return [float(x) for x in scores.detach().cpu().tolist()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO baseline with the same scalar reward model.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--reward_model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--dataset_kind", choices=["uf_binarized", "uf_listwise", "nectar"], default="uf_binarized")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--split", default="train_prefs")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.01, help="KL coefficient to reference model.")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--scale_rewards", choices=["group", "batch", "none"], default="group")
    parser.add_argument("--loss_type", choices=["grpo", "dapo", "dr_grpo", "bnpo"], default="dapo")
    parser.add_argument("--mask_truncated_completions", action="store_true")
    parser.add_argument("--reward_max_length", type=int, default=1024)
    parser.add_argument("--reward_batch_size", type=int, default=8)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--reward_shift", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--log_completions", action="store_true")

    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto", "none"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default=None)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_mode", choices=["colocate", "server"], default="colocate")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--report_to", default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    train_dataset = load_prompt_dataset(
        dataset_kind=args.dataset_kind,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        split=args.split,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    tokenizer = load_tokenizer(args.model_name_or_path, padding_side="left", trust_remote_code=args.trust_remote_code)
    reward_func = RewardModelScorer(
        args.reward_model_name_or_path,
        dtype=args.dtype,
        max_length=args.reward_max_length,
        batch_size=args.reward_batch_size,
        reward_scale=args.reward_scale,
        reward_shift=args.reward_shift,
        trust_remote_code=args.trust_remote_code,
    )

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

    model_init_kwargs = {"trust_remote_code": args.trust_remote_code}
    dtype = pick_dtype(args.dtype)
    if dtype is not None:
        model_init_kwargs["torch_dtype"] = dtype

    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        beta=args.beta,
        epsilon=args.epsilon,
        num_iterations=args.num_iterations,
        scale_rewards=args.scale_rewards,
        loss_type=args.loss_type,
        mask_truncated_completions=args.mask_truncated_completions,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps" if args.save_steps > 0 else "no",
        log_completions=args.log_completions,
        bf16=args.dtype in {"bf16", "bfloat16"},
        fp16=args.dtype in {"fp16", "float16"},
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        remove_unused_columns=False,
        model_init_kwargs=model_init_kwargs,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )

    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        args=grpo_args,
        train_dataset=train_dataset,
        reward_funcs=reward_func,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    save_json(output_dir / "grpo_run_config.json", vars(args))


if __name__ == "__main__":
    main()
