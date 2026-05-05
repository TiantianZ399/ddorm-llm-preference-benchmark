#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
SEED="${1:-42}"
MODEL="${MODEL:-EleutherAI/pythia-410m}"
RM="${RM:-outputs/gpu/reward_pythia410m_seed${SEED}}"
K="${K:-4}"
OUT="${OUT:-outputs/gpu/grpo_k${K}_same_rm_seed${SEED}}"

accelerate launch src/benchmark/gpu/train_grpo_trl.py \
  --model_name_or_path "$MODEL" \
  --reward_model_name_or_path "$RM" \
  --output_dir "$OUT" \
  --dataset_kind uf_listwise \
  --split train \
  --max_examples "${MAX_EXAMPLES:-20000}" \
  --seed "$SEED" \
  --learning_rate 1e-6 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --num_generations "$K" \
  --max_completion_length 256 \
  --temperature 1.0 \
  --beta 0.01 \
  --scale_rewards group \
  --loss_type dapo \
  --dtype bf16 \
  --gradient_checkpointing \
  --use_lora \
  --logging_steps 10

python -m benchmark.gpu.evaluate_listwise_gpu \
  --model_name_or_path "$OUT" \
  --base_model_for_lora "$MODEL" \
  --output_json "$OUT/eval_listwise_k${K}.json" \
  --dataset_kind uf_listwise \
  --split train \
  --max_candidates "$K" \
  --candidate_strategy random_include_best \
  --max_examples 2000 \
  --batch_size 2 \
  --max_length 1024 \
  --dtype bf16 \
  --seed "$SEED"
