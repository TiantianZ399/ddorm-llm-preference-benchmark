#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
SEED="${1:-42}"
MODEL="${MODEL:-EleutherAI/pythia-410m}"
OUT="${OUT:-outputs/gpu/reward_pythia410m_seed${SEED}}"

accelerate launch src/benchmark/gpu/train_reward_gpu.py \
  --base_model "$MODEL" \
  --output_dir "$OUT" \
  --dataset_name HuggingFaceH4/ultrafeedback_binarized \
  --train_split train_prefs \
  --eval_split test_prefs \
  --seed "$SEED" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --max_length 1024 \
  --dtype bf16 \
  --gradient_checkpointing \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --logging_steps 10
