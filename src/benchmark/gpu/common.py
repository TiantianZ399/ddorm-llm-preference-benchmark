from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_dtype(dtype: str = "bf16") -> torch.dtype | None:
    dtype = dtype.lower()
    if dtype in {"none", "auto"}:
        return None
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_tokenizer(
    model_name_or_path: str,
    *,
    padding_side: str = "right",
    trust_remote_code: bool = False,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        padding_side=padding_side,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def infer_lora_targets(model_name_or_path: str) -> list[str]:
    lower = model_name_or_path.lower()
    if any(key in lower for key in ["llama", "mistral", "qwen", "gemma", "deepseek"]):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if any(key in lower for key in ["pythia", "gpt-neox"]):
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    if "gpt2" in lower:
        return ["c_attn", "c_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def build_lora_config(
    model_name_or_path: str,
    *,
    task_type: str,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: str | None = None,
):
    from peft import LoraConfig, TaskType

    if task_type.upper() in {"CAUSAL_LM", "LM"}:
        peft_task = TaskType.CAUSAL_LM
    elif task_type.upper() in {"SEQ_CLS", "SEQUENCE_CLASSIFICATION", "REWARD"}:
        peft_task = TaskType.SEQ_CLS
    else:
        raise ValueError(f"Unsupported LoRA task_type: {task_type}")

    targets = None
    if target_modules:
        targets = [item.strip() for item in target_modules.split(",") if item.strip()]
    else:
        targets = infer_lora_targets(model_name_or_path)

    return LoraConfig(
        task_type=peft_task,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
    )


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def append_jsonl(path: str | os.PathLike[str], row: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def save_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


def count_trainable_parameters(model: torch.nn.Module) -> dict[str, int | float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = float(trainable / total) if total else 0.0
    return {"trainable": trainable, "total": total, "ratio": ratio}
