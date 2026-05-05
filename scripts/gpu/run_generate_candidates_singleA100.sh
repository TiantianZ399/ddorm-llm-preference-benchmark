#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
SEED="${1:-42}"
K="${K:-8}"
MODEL="${MODEL:-EleutherAI/pythia-410m}"
RM="${RM:-outputs/gpu/reward_pythia410m_seed${SEED}}"
OUT="${OUT:-outputs/gpu/generated_candidates_k${K}_seed${SEED}}"

python -m benchmark.gpu.generate_candidates_gpu \
  --model_name_or_path "$MODEL" \
  --reward_model_name_or_path "$RM" \
  --output_dataset_path "$OUT" \
  --dataset_kind uf_binarized \
  --split train_prefs \
  --max_examples "${MAX_EXAMPLES:-20000}" \
  --num_candidates "$K" \
  --per_device_batch_size 2 \
  --max_prompt_length 768 \
  --max_new_tokens 256 \
  --generation_temperature 0.8 \
  --top_p 0.95 \
  --dtype bf16 \
  --seed "$SEED"
