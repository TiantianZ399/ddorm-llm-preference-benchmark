from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from .common import ensure_dir, load_tokenizer, pick_dtype, save_json, set_seed
from .data import CandidateCollator, load_candidate_dataset
from .scoring import flatten_prompt_candidates, sequence_logprobs


def ndcg_at_k(scores: list[float], labels: list[float], k: int | None = None) -> float:
    if k is None:
        k = len(labels)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    ideal = sorted(range(len(labels)), key=lambda i: labels[i], reverse=True)[:k]

    def dcg(indices: list[int]) -> float:
        return sum((2.0 ** labels[i] - 1.0) / math.log2(rank + 2.0) for rank, i in enumerate(indices))

    denom = dcg(ideal)
    return float(dcg(order) / denom) if denom > 0 else 0.0


def pairwise_accuracy(scores: list[float], labels: list[float]) -> float:
    correct = 0
    total = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if labels[i] == labels[j]:
                continue
            total += 1
            correct += int((scores[i] - scores[j]) * (labels[i] - labels[j]) > 0)
    return float(correct / total) if total else float("nan")


def spearman_no_scipy(scores: list[float], labels: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for rank, idx in enumerate(order):
            out[idx] = float(rank)
        return out

    rx = ranks(scores)
    ry = ranks(labels)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    deny = math.sqrt(sum((y - my) ** 2 for y in ry))
    return float(num / (denx * deny)) if denx > 0 and deny > 0 else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Listwise K>2 evaluation for finite-candidate methods.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--base_model_for_lora", default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--dataset_kind", choices=["uf_listwise", "nectar"], default="uf_listwise")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_path", default=None, help="Optional load_from_disk candidate dataset path.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--score_mode", default="mean_rating")
    parser.add_argument("--max_candidates", type=int, default=4)
    parser.add_argument("--candidate_strategy", choices=["first", "top_gold", "random", "random_include_best"], default="first")
    parser.add_argument("--max_examples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
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
        dataset_path=args.dataset_path,
        split=args.split,
        score_mode=args.score_mode,
        max_candidates=args.max_candidates,
        candidate_strategy=args.candidate_strategy,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=CandidateCollator())

    top1 = []
    ndcgs = []
    pair_accs = []
    spearmans = []
    mean_margins_best_second = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="listwise eval"):
            prompts = batch["prompts"]
            candidates = batch["candidates"]
            rewards = batch["reward_scores"]
            bsz = len(prompts)
            k = len(candidates[0])
            flat_prompts, flat_candidates = flatten_prompt_candidates(prompts, candidates)
            scores = sequence_logprobs(
                model,
                tokenizer,
                flat_prompts,
                flat_candidates,
                max_length=args.max_length,
                add_eos=True,
                normalize_by_length=True,
            ).avg_logprob.view(bsz, k).detach().float().cpu().tolist()
            for pred, gold in zip(scores, rewards):
                pred_best = max(range(len(pred)), key=lambda i: pred[i])
                gold_best = max(range(len(gold)), key=lambda i: gold[i])
                top1.append(float(pred_best == gold_best))
                ndcgs.append(ndcg_at_k(pred, gold))
                pair_accs.append(pairwise_accuracy(pred, gold))
                spearmans.append(spearman_no_scipy(pred, gold))
                ordered = sorted(pred, reverse=True)
                if len(ordered) >= 2:
                    mean_margins_best_second.append(float(ordered[0] - ordered[1]))

    def mean_valid(xs: list[float]) -> float:
        vals = [x for x in xs if not math.isnan(x)]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    metrics = {
        "n": len(top1),
        "k": args.max_candidates,
        "top1_accuracy_vs_gold": mean_valid(top1),
        "ndcg": mean_valid(ndcgs),
        "pairwise_accuracy_vs_gold": mean_valid(pair_accs),
        "spearman_vs_gold": mean_valid(spearmans),
        "mean_best_second_policy_margin": mean_valid(mean_margins_best_second),
        "model_name_or_path": args.model_name_or_path,
        "dataset_kind": args.dataset_kind,
        "dataset_name": args.dataset_name,
        "split": args.split,
    }
    ensure_dir(Path(args.output_json).parent)
    save_json(args.output_json, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
