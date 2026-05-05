from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

from .common import ensure_dir, load_tokenizer, pick_dtype, save_json, set_seed
from .data import load_prompt_dataset
from .scoring import reward_model_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate K candidates per prompt and score them with a reward model.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--reward_model_name_or_path", required=True)
    parser.add_argument("--output_dataset_path", required=True)
    parser.add_argument("--base_model_for_lora", default=None)

    parser.add_argument("--dataset_kind", choices=["uf_binarized", "uf_listwise", "nectar"], default="uf_binarized")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--split", default="train_prefs")
    parser.add_argument("--max_examples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_candidates", type=int, default=8)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--generation_temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--reward_max_length", type=int, default=1024)
    parser.add_argument("--reward_batch_size", type=int, default=8)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto", "none"], default="bf16")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def load_policy(args: argparse.Namespace, tokenizer):
    dtype = pick_dtype(args.dtype)
    if args.base_model_for_lora:
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            args.base_model_for_lora,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        model = PeftModel.from_pretrained(base, args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_path = Path(args.output_dataset_path)
    ensure_dir(out_path.parent)

    tokenizer_source = args.base_model_for_lora or args.model_name_or_path
    tokenizer = load_tokenizer(tokenizer_source, padding_side="left", trust_remote_code=args.trust_remote_code)
    reward_tokenizer = load_tokenizer(args.reward_model_name_or_path, padding_side="right", trust_remote_code=args.trust_remote_code)

    model = load_policy(args, tokenizer)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name_or_path,
        num_labels=1,
        torch_dtype=pick_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    reward_model.to(device).eval()

    prompt_ds = load_prompt_dataset(
        dataset_kind=args.dataset_kind,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        split=args.split,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    dataloader = DataLoader(prompt_ds, batch_size=args.per_device_batch_size, shuffle=False)

    rows: list[dict] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="generating candidates"):
            prompts = [str(x) for x in batch["prompt"]]
            enc = tokenizer(
                prompts,
                truncation=True,
                max_length=args.max_prompt_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            generated = model.generate(
                **enc,
                do_sample=True,
                num_return_sequences=args.num_candidates,
                max_new_tokens=args.max_new_tokens,
                temperature=args.generation_temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            context_len = enc["input_ids"].shape[1]
            completions = tokenizer.batch_decode(generated[:, context_len:], skip_special_tokens=True)
            grouped: list[list[str]] = []
            for i in range(len(prompts)):
                start = i * args.num_candidates
                grouped.append([c.strip() for c in completions[start : start + args.num_candidates]])

            flat_prompts = [p for p in prompts for _ in range(args.num_candidates)]
            flat_completions = [c for group in grouped for c in group]
            rewards = reward_model_scores(
                reward_model,
                reward_tokenizer,
                flat_prompts,
                flat_completions,
                max_length=args.reward_max_length,
                batch_size=args.reward_batch_size,
            ).view(len(prompts), args.num_candidates)

            for prompt, cand_list, reward_vec in zip(prompts, grouped, rewards.detach().float().cpu().tolist()):
                rows.append(
                    {
                        "prompt": prompt,
                        "candidates": cand_list,
                        "reward_scores": [float(x) for x in reward_vec],
                        "prompt_id": str(len(rows)),
                    }
                )

    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(out_path))
    save_json(
        out_path / "generation_config.json",
        {
            **vars(args),
            "num_rows": len(rows),
            "num_candidates": args.num_candidates,
        },
    )
    print(f"Saved {len(rows)} prompts with K={args.num_candidates} candidates to {out_path}")


if __name__ == "__main__":
    main()
