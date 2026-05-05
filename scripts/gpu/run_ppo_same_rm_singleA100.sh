#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
SEED="${1:-42}"
MODEL="${MODEL:-EleutherAI/pythia-410m}"
RM="${RM:-outputs/gpu/reward_pythia410m_seed${SEED}}"
OUT="${OUT:-outputs/gpu/ppo_same_rm_seed${SEED}}"

accelerate launch src/benchmark/gpu/train_ppo_trl.py \
  --model_name_or_path "$MODEL" \
  --reward_model_name_or_path "$RM" \
  --output_dir "$OUT" \
  --dataset_kind uf_binarized \
  --split train_prefs \
  --seed "$SEED" \
  --learning_rate 3e-6 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_ppo_epochs 4 \
  --num_mini_batches 1 \
  --local_rollout_forward_batch_size 4 \
  --response_length 256 \
  --max_prompt_length 768 \
  --temperature 0.7 \
  --kl_coef 0.05 \
  --missing_eos_penalty 1.0 \
  --dtype bf16 \
  --gradient_checkpointing \
  --use_lora \
  --logging_steps 10

python -m benchmark.gpu.evaluate_pairwise_gpu \
  --model_name_or_path "$OUT" \
  --base_model_for_lora "$MODEL" \
  --output_json "$OUT/eval_pairwise_test_prefs.json" \
  --split test_prefs \
  --batch_size 4 \
  --max_length 1024 \
  --dtype bf16 \
  --seed "$SEED"
