#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
SEED="${1:-42}"
MODEL="${MODEL:-EleutherAI/pythia-410m}"
RM="${RM:-outputs/gpu/reward_pythia410m_seed${SEED}}"

# K=2: original binarized chosen/rejected setup.
OUT="outputs/gpu/ddorm_k2_seed${SEED}"
accelerate launch src/benchmark/gpu/train_ddorm_gpu.py \
  --model_name_or_path "$MODEL" \
  --reward_model_name_or_path "$RM" \
  --output_dir "$OUT" \
  --dataset_kind uf_binarized \
  --split train_prefs \
  --seed "$SEED" \
  --eta "${ETA:-1.0}" \
  --decision_temperature "${TAU:-1.0}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-6 \
  --max_length 1024 \
  --dtype bf16 \
  --gradient_checkpointing \
  --use_lora
python -m benchmark.gpu.evaluate_pairwise_gpu \
  --model_name_or_path "$OUT" \
  --base_model_for_lora "$MODEL" \
  --output_json "$OUT/eval_pairwise_test_prefs.json" \
  --split test_prefs \
  --batch_size 4 \
  --max_length 1024 \
  --dtype bf16 \
  --seed "$SEED"

# K>2: generated finite candidate pools scored by the same reward model.
for K in 4 8 16; do
  CAND="outputs/gpu/generated_candidates_k${K}_seed${SEED}"
  if [[ ! -d "$CAND" ]]; then
    K="$K" OUT="$CAND" bash scripts/gpu/run_generate_candidates_singleA100.sh "$SEED"
  fi
  OUT="outputs/gpu/ddorm_k${K}_seed${SEED}"
  accelerate launch src/benchmark/gpu/train_ddorm_gpu.py \
    --model_name_or_path "$MODEL" \
    --reward_model_name_or_path "$RM" \
    --output_dir "$OUT" \
    --dataset_kind uf_binarized \
    --dataset_path "$CAND" \
    --split train_prefs \
    --seed "$SEED" \
    --eta "${ETA:-1.0}" \
    --decision_temperature "${TAU:-1.0}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --max_length 1024 \
    --dtype bf16 \
    --gradient_checkpointing \
    --use_lora
  python -m benchmark.gpu.evaluate_listwise_gpu \
    --model_name_or_path "$OUT" \
    --base_model_for_lora "$MODEL" \
    --output_json "$OUT/eval_generated_k${K}.json" \
    --dataset_kind uf_listwise \
    --dataset_path "$CAND" \
    --split train \
    --max_candidates "$K" \
    --max_examples 2000 \
    --batch_size 2 \
    --max_length 1024 \
    --dtype bf16 \
    --seed "$SEED"
done
python -m benchmark.gpu.summarize_gpu_results --input_glob 'outputs/gpu/**/eval*.json' --output_csv outputs/gpu/results_summary.csv
