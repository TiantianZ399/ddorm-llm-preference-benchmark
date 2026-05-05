from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class SequenceScoreOutput:
    avg_logprob: torch.Tensor
    sum_logprob: torch.Tensor
    lengths: torch.Tensor


def _encode_no_special(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def build_prompt_completion_tensors(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    completions: list[str],
    *,
    max_length: int = 1024,
    add_eos: bool = True,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Build padded LM inputs and labels that mask prompt tokens.

    Labels are -100 on prompt/pad tokens and equal to token ids on completion tokens,
    so shifted causal-LM logprobs score only completions.
    """
    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have same length")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be set")

    eos_id = tokenizer.eos_token_id
    rows: list[list[int]] = []
    labels: list[list[int]] = []
    for prompt, completion in zip(prompts, completions):
        prompt_ids = _encode_no_special(tokenizer, prompt)
        completion_ids = _encode_no_special(tokenizer, completion)
        if add_eos and eos_id is not None:
            completion_ids = completion_ids + [int(eos_id)]
        ids = prompt_ids + completion_ids
        lab = [-100] * len(prompt_ids) + completion_ids
        if len(ids) > max_length:
            # Keep the rightmost context because the completion is what we score.
            overflow = len(ids) - max_length
            ids = ids[overflow:]
            lab = lab[overflow:]
            # If truncation removed the whole prompt and part of the completion, lab remains valid.
        rows.append(ids)
        labels.append(lab)

    max_len = max(len(x) for x in rows)
    input_ids = []
    attention_mask = []
    label_ids = []
    for ids, lab in zip(rows, labels):
        pad_len = max_len - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        label_ids.append(lab + [-100] * pad_len)

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(label_ids, dtype=torch.long, device=device),
    }
    return batch


def sequence_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    completions: list[str],
    *,
    max_length: int = 1024,
    add_eos: bool = True,
    normalize_by_length: bool = True,
) -> SequenceScoreOutput:
    """Return differentiable sequence log-probability scores for completions."""
    device = next(model.parameters()).device
    batch = build_prompt_completion_tensors(
        tokenizer,
        prompts,
        completions,
        max_length=max_length,
        add_eos=add_eos,
        device=device,
    )
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    token_logprobs = F.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * mask
    lengths = mask.sum(dim=-1).clamp_min(1)
    sum_logprob = token_logprobs.sum(dim=-1)
    avg_logprob = sum_logprob / lengths if normalize_by_length else sum_logprob
    return SequenceScoreOutput(avg_logprob=avg_logprob, sum_logprob=sum_logprob, lengths=lengths)


@torch.no_grad()
def reward_model_scores(
    reward_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    completions: list[str],
    *,
    max_length: int = 1024,
    batch_size: int = 8,
) -> torch.Tensor:
    """Score prompt-completion pairs with a scalar sequence-classification reward model."""
    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have same length")
    device = next(reward_model.parameters()).device
    scores: list[torch.Tensor] = []
    texts = [p + c for p, c in zip(prompts, completions)]
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        batch = tokenizer(
            chunk,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        outputs = reward_model(**batch)
        logits = outputs.logits
        if logits.ndim == 2 and logits.shape[-1] == 1:
            score = logits[:, 0]
        elif logits.ndim == 2:
            score = logits[:, -1]
        else:
            score = logits.reshape(logits.shape[0], -1)[:, -1]
        scores.append(score.detach())
    return torch.cat(scores, dim=0)


def flatten_prompt_candidates(prompts: list[str], candidates: list[list[str]]) -> tuple[list[str], list[str]]:
    flat_prompts: list[str] = []
    flat_candidates: list[str] = []
    for prompt, cand_list in zip(prompts, candidates):
        for candidate in cand_list:
            flat_prompts.append(prompt)
            flat_candidates.append(candidate)
    return flat_prompts, flat_candidates


def manual_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    """Small dependency-free ROC-AUC fallback for binary labels."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(int(y == 1) for _, y in pairs)
    neg = sum(int(y == 0) for _, y in pairs)
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = 0.0
    for rank, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum += rank
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)
