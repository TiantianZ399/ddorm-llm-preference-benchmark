from __future__ import annotations

import random
import statistics
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk

USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
SYSTEM_TAG = "<|system|>"
OTHER_TAG = "<|other|>"


def _role_tag(role: str) -> str:
    role = str(role).lower().strip()
    if role == "user":
        return USER_TAG
    if role == "assistant":
        return ASSISTANT_TAG
    if role == "system":
        return SYSTEM_TAG
    return f"{OTHER_TAG}:{role}"


def render_messages(messages: list[dict[str, Any]], *, add_assistant_prefix: bool = False) -> str:
    chunks: list[str] = []
    for msg in messages:
        role = _role_tag(str(msg.get("role", "user")))
        content = str(msg.get("content", "")).strip()
        chunks.append(f"{role}\n{content}\n")
    if add_assistant_prefix:
        if not messages or str(messages[-1].get("role", "")).lower() != "assistant":
            chunks.append(f"{ASSISTANT_TAG}\n")
    return "\n".join(chunks).strip() + ("\n" if chunks else "")


def split_prompt_and_completion(messages: list[dict[str, Any]]) -> tuple[str, str]:
    if len(messages) < 2:
        raise ValueError("Expected prompt messages plus one assistant completion.")
    prompt_messages = messages[:-1]
    completion_message = messages[-1]
    if str(completion_message.get("role", "")).lower() != "assistant":
        raise ValueError("Last message must be an assistant completion.")
    prompt = render_messages(prompt_messages, add_assistant_prefix=True)
    completion = str(completion_message.get("content", "")).strip()
    return prompt, completion


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "None":
            return None
        return float(value)
    except Exception:
        return None


def ultrafeedback_completion_score(completion: dict[str, Any], *, mode: str = "mean_rating") -> float:
    annotations = completion.get("annotations", {}) or {}
    if mode == "mean_rating":
        ratings: list[float] = []
        for aspect in annotations.values():
            if isinstance(aspect, dict):
                rating = _safe_float(aspect.get("Rating"))
                if rating is not None:
                    ratings.append(rating)
        if ratings:
            return float(statistics.mean(ratings))
        return 0.0
    if mode == "helpfulness":
        rating = _safe_float(annotations.get("helpfulness", {}).get("Rating"))
        return 0.0 if rating is None else float(rating)
    if mode == "overall":
        for key in ["overall", "Overall", "score", "Score"]:
            rating = _safe_float(annotations.get(key, {}).get("Rating") if isinstance(annotations.get(key), dict) else annotations.get(key))
            if rating is not None:
                return float(rating)
        return ultrafeedback_completion_score(completion, mode="mean_rating")
    raise ValueError(f"Unknown UltraFeedback score mode: {mode}")


def select_candidates(
    candidates: list[str],
    reward_scores: list[float],
    *,
    max_candidates: int | None = None,
    strategy: str = "first",
    seed: int = 0,
    row_index: int = 0,
) -> tuple[list[str], list[float]]:
    if len(candidates) != len(reward_scores):
        raise ValueError("candidates and reward_scores must have same length")
    pairs = list(zip(candidates, reward_scores))
    if max_candidates is None or max_candidates <= 0 or len(pairs) <= max_candidates:
        return list(candidates), [float(x) for x in reward_scores]

    if strategy == "first":
        selected = pairs[:max_candidates]
    elif strategy == "top_gold":
        selected = sorted(pairs, key=lambda z: z[1], reverse=True)[:max_candidates]
    elif strategy == "random":
        rng = random.Random(seed + 1009 * row_index)
        selected = rng.sample(pairs, max_candidates)
    elif strategy == "random_include_best":
        best = max(pairs, key=lambda z: z[1])
        rest = [p for p in pairs if p is not best]
        rng = random.Random(seed + 1009 * row_index)
        selected = [best] + rng.sample(rest, max_candidates - 1)
        rng.shuffle(selected)
    else:
        raise ValueError(f"Unknown candidate selection strategy: {strategy}")
    return [x for x, _ in selected], [float(y) for _, y in selected]


def convert_h4_binarized_pref(example: dict[str, Any], idx: int = 0) -> dict[str, Any]:
    chosen_obj = example["chosen"]
    rejected_obj = example["rejected"]
    if isinstance(chosen_obj, list):
        prompt_chosen, chosen = split_prompt_and_completion(chosen_obj)
        prompt_rejected, rejected = split_prompt_and_completion(rejected_obj)
        if prompt_chosen != prompt_rejected:
            raise ValueError("Chosen and rejected prompts do not match.")
        prompt = prompt_chosen
    else:
        prompt = str(example.get("prompt", ""))
        chosen = str(chosen_obj)
        rejected = str(rejected_obj)
    return {
        "prompt": prompt,
        "candidates": [chosen, rejected],
        "reward_scores": [float(example.get("score_chosen", 1.0)), float(example.get("score_rejected", 0.0))],
        "prompt_id": str(example.get("prompt_id", idx)),
    }


def load_ultrafeedback_binarized_candidates(
    *,
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
    split: str = "train_prefs",
    max_examples: int | None = None,
    seed: int = 42,
) -> Dataset:
    ds = load_dataset(dataset_name, split=split)
    if max_examples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(max_examples, len(ds))))
    return ds.map(convert_h4_binarized_pref, with_indices=True, remove_columns=ds.column_names)


def load_openbmb_ultrafeedback_listwise(
    *,
    dataset_name: str = "openbmb/UltraFeedback",
    split: str = "train",
    score_mode: str = "mean_rating",
    max_candidates: int | None = 4,
    candidate_strategy: str = "first",
    max_examples: int | None = None,
    seed: int = 42,
) -> Dataset:
    ds = load_dataset(dataset_name, split=split)
    ds = ds.shuffle(seed=seed)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    def _convert(example: dict[str, Any], idx: int) -> dict[str, Any]:
        instruction = str(example.get("instruction", example.get("prompt", ""))).strip()
        prompt = f"{USER_TAG}\n{instruction}\n\n{ASSISTANT_TAG}\n"
        candidates: list[str] = []
        scores: list[float] = []
        model_names: list[str] = []
        for comp in example.get("completions", []):
            text = comp.get("response", comp.get("completion", comp.get("text", "")))
            candidates.append(str(text).strip())
            scores.append(float(ultrafeedback_completion_score(comp, mode=score_mode)))
            model_names.append(str(comp.get("model", "unknown")))
        candidates, scores = select_candidates(
            candidates,
            scores,
            max_candidates=max_candidates,
            strategy=candidate_strategy,
            seed=seed,
            row_index=idx,
        )
        return {
            "prompt": prompt,
            "candidates": candidates,
            "reward_scores": scores,
            "models": model_names[: len(candidates)],
            "prompt_id": str(example.get("id", idx)),
        }

    out = ds.map(_convert, with_indices=True, remove_columns=ds.column_names)
    return out.filter(lambda ex: len(ex["candidates"]) >= 2 and len(ex["candidates"]) == len(ex["reward_scores"]))


def load_nectar_listwise(
    *,
    dataset_name: str = "berkeley-nest/Nectar",
    split: str = "train",
    max_candidates: int | None = 7,
    candidate_strategy: str = "first",
    max_examples: int | None = None,
    seed: int = 42,
) -> Dataset:
    ds = load_dataset(dataset_name, split=split)
    ds = ds.shuffle(seed=seed)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    def _convert(example: dict[str, Any], idx: int) -> dict[str, Any]:
        prompt = str(example.get("prompt", "")).rstrip()
        if not prompt.endswith("Assistant:"):
            prompt = prompt + "\n\nAssistant:"
        answers = example.get("answers", [])
        candidates = [str(ans.get("answer", "")).strip() for ans in answers]
        # Nectar ranks are ordinal; lower rank is better. Negating makes larger reward better.
        scores = [float(-int(ans.get("rank", i + 1))) for i, ans in enumerate(answers)]
        candidates, scores = select_candidates(
            candidates,
            scores,
            max_candidates=max_candidates,
            strategy=candidate_strategy,
            seed=seed,
            row_index=idx,
        )
        return {
            "prompt": prompt,
            "candidates": candidates,
            "reward_scores": scores,
            "prompt_id": str(example.get("id", idx)),
        }

    out = ds.map(_convert, with_indices=True, remove_columns=ds.column_names)
    return out.filter(lambda ex: len(ex["candidates"]) >= 2 and len(ex["candidates"]) == len(ex["reward_scores"]))


def load_candidate_dataset(
    *,
    dataset_kind: str,
    dataset_name: str | None = None,
    split: str,
    score_mode: str = "mean_rating",
    max_candidates: int | None = None,
    candidate_strategy: str = "first",
    max_examples: int | None = None,
    seed: int = 42,
    dataset_path: str | None = None,
) -> Dataset:
    """Load a finite-candidate dataset with columns prompt, candidates, reward_scores."""
    if dataset_path:
        ds = load_from_disk(dataset_path)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        return ds

    if dataset_kind == "uf_binarized":
        return load_ultrafeedback_binarized_candidates(
            dataset_name=dataset_name or "HuggingFaceH4/ultrafeedback_binarized",
            split=split,
            max_examples=max_examples,
            seed=seed,
        )
    if dataset_kind == "uf_listwise":
        return load_openbmb_ultrafeedback_listwise(
            dataset_name=dataset_name or "openbmb/UltraFeedback",
            split=split,
            score_mode=score_mode,
            max_candidates=max_candidates,
            candidate_strategy=candidate_strategy,
            max_examples=max_examples,
            seed=seed,
        )
    if dataset_kind == "nectar":
        return load_nectar_listwise(
            dataset_name=dataset_name or "berkeley-nest/Nectar",
            split=split,
            max_candidates=max_candidates,
            candidate_strategy=candidate_strategy,
            max_examples=max_examples,
            seed=seed,
        )
    raise ValueError(f"Unknown dataset_kind: {dataset_kind}")


def candidate_dataset_to_prompts(ds: Dataset) -> Dataset:
    cols = ds.column_names
    remove_cols = [c for c in cols if c != "prompt"]
    out = ds.remove_columns(remove_cols) if remove_cols else ds
    # Deduplicate prompts while preserving order.
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for row in out:
        prompt = str(row["prompt"])
        if prompt not in seen:
            seen.add(prompt)
            rows.append({"prompt": prompt})
    return Dataset.from_list(rows)


def load_prompt_dataset(
    *,
    dataset_kind: str,
    dataset_name: str | None = None,
    split: str,
    max_examples: int | None = None,
    seed: int = 42,
    dataset_path: str | None = None,
) -> Dataset:
    if dataset_kind in {"uf_binarized", "uf_listwise", "nectar"} or dataset_path:
        candidates = load_candidate_dataset(
            dataset_kind=dataset_kind,
            dataset_name=dataset_name,
            split=split,
            max_examples=max_examples,
            seed=seed,
            dataset_path=dataset_path,
        )
        return candidate_dataset_to_prompts(candidates)
    raise ValueError(f"Unknown prompt dataset_kind: {dataset_kind}")


class CandidateCollator:
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Empty batch")
        k = len(features[0]["candidates"])
        if any(len(f["candidates"]) != k for f in features):
            raise ValueError("All examples in a DDO-RM batch must have the same number of candidates.")
        return {
            "prompts": [str(f["prompt"]) for f in features],
            "candidates": [[str(c) for c in f["candidates"]] for f in features],
            "reward_scores": [[float(x) for x in f.get("reward_scores", [])] for f in features],
            "prompt_ids": [str(f.get("prompt_id", i)) for i, f in enumerate(features)],
        }
