from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from .common import ensure_dir, load_tokenizer, pick_dtype, save_json, set_seed
from .data import CandidateCollator, load_candidate_dataset
from .scoring import flatten_prompt_candidates, manual_auc, sequence_logprobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise held-out evaluation for DDO-RM/PPO/GRPO/DPO models.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--base_model_for_lora", default=None, help="Base model path if model_name_or_path is a PEFT adapter.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--dataset_kind", choices=["uf_binarized"], default="uf_binarized")
    parser.add_argument("--dataset_name", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--split", default="test_prefs")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto", "none"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def load_policy(args: argparse.Namespace):
    dtype = pick_dtype(args.dtype)
    tokenizer_source = args.model_name_or_path if args.base_model_for_lora is None else args.base_model_for_lora
    tokenizer = load_tokenizer(tokenizer_source, padding_side="right", trust_remote_code=args.trust_remote_code)
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
    return model, tokenizer


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    model, tokenizer = load_policy(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = load_candidate_dataset(
        dataset_kind=args.dataset_kind,
        dataset_name=args.dataset_name,
        split=args.split,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=CandidateCollator())

    margins: list[float] = []
    chosen_scores: list[float] = []
    rejected_scores: list[float] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="pairwise eval"):
            prompts = batch["prompts"]
            candidates = batch["candidates"]
            if len(candidates[0]) != 2:
                raise ValueError("Pairwise evaluation requires exactly two candidates per prompt.")
            flat_prompts, flat_candidates = flatten_prompt_candidates(prompts, candidates)
            scores = sequence_logprobs(
                model,
                tokenizer,
                flat_prompts,
                flat_candidates,
                max_length=args.max_length,
                add_eos=True,
                normalize_by_length=True,
            ).avg_logprob.view(len(prompts), 2)
            chosen = scores[:, 0].detach().float().cpu()
            rejected = scores[:, 1].detach().float().cpu()
            chosen_scores.extend(chosen.tolist())
            rejected_scores.extend(rejected.tolist())
            margins.extend((chosen - rejected).tolist())

    n = len(margins)
    labels = [1] * n + [0] * n
    auc_scores = chosen_scores + rejected_scores
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(labels, auc_scores))
    except Exception:
        auc = float(manual_auc(labels, auc_scores))

    metrics = {
        "n": n,
        "pair_accuracy": float(sum(m > 0 for m in margins) / max(n, 1)),
        "auc": auc,
        "mean_margin": float(sum(margins) / max(n, 1)),
        "mean_chosen_score": float(sum(chosen_scores) / max(n, 1)),
        "mean_rejected_score": float(sum(rejected_scores) / max(n, 1)),
        "model_name_or_path": args.model_name_or_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
    }
    ensure_dir(Path(args.output_json).parent)
    save_json(args.output_json, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
