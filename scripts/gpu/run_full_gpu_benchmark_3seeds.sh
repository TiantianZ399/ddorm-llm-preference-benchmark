#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
SEEDS=(42 13 3407)
for SEED in "${SEEDS[@]}"; do
  bash scripts/gpu/run_reward_pythia410m_single4090.sh "$SEED"
  bash scripts/gpu/run_ddorm_k_sweep_singleA100.sh "$SEED"
  bash scripts/gpu/run_ppo_same_rm_singleA100.sh "$SEED"
  K=4 bash scripts/gpu/run_grpo_same_rm_singleA100.sh "$SEED"
  K=8 bash scripts/gpu/run_grpo_same_rm_singleA100.sh "$SEED"
done
python -m benchmark.gpu.summarize_gpu_results --input_glob 'outputs/gpu/**/eval*.json' --output_csv outputs/gpu/results_summary.csv
