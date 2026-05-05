# GPU experiment overlay for DDO-RM

This overlay adds the GPU code needed for the next empirical stage of the DDO-RM project:

1. train a scalar reward model;
2. run DDO-RM with the same reward model;
3. run a matched PPO-RLHF baseline with the same reward model;
4. run GRPO with the same reward model for multi-candidate / group-generation settings;
5. evaluate pairwise and listwise metrics;
6. log DDO-RM diagnostics such as realized `KL(q || p)`, target entropy, reward gain, and `eta / tau`.

The files are designed to be copied into the existing repository root.

## Added files

```text
requirements-gpu.txt
README_GPU_EXPERIMENTS.md
configs/accelerate/single_gpu.yaml
configs/accelerate/deepspeed_zero2.yaml
configs/accelerate/deepspeed_zero3.yaml
configs/gpu/*.example.env
scripts/gpu/install_gpu_env.sh
scripts/gpu/run_reward_pythia410m_single4090.sh
scripts/gpu/run_ddorm_k2_single4090.sh
scripts/gpu/run_ddorm_k_sweep_singleA100.sh
scripts/gpu/run_ppo_same_rm_singleA100.sh
scripts/gpu/run_grpo_same_rm_singleA100.sh
scripts/gpu/run_full_gpu_benchmark_3seeds.sh
src/benchmark/gpu/common.py
src/benchmark/gpu/data.py
src/benchmark/gpu/ddorm_math.py
src/benchmark/gpu/scoring.py
src/benchmark/gpu/train_reward_gpu.py
src/benchmark/gpu/generate_candidates_gpu.py
src/benchmark/gpu/train_ddorm_gpu.py
src/benchmark/gpu/train_ppo_trl.py
src/benchmark/gpu/train_grpo_trl.py
src/benchmark/gpu/evaluate_pairwise_gpu.py
src/benchmark/gpu/evaluate_listwise_gpu.py
src/benchmark/gpu/summarize_gpu_results.py
tests/test_ddorm_math.py
```

## Environment

Use a fresh environment because the PPO/GRPO scripts target the current TRL v1+ API.

```bash
python -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
pip install -e . || true
export PYTHONPATH="${PYTHONPATH:-}:src"
```

## Minimal matched-reward-model experiment

Train the reward model:

```bash
bash scripts/gpu/run_reward_pythia410m_single4090.sh 42
```

Run DDO-RM on the binarized K=2 setup:

```bash
bash scripts/gpu/run_ddorm_k2_single4090.sh 42
```

Run PPO-RLHF with the same reward model:

```bash
bash scripts/gpu/run_ppo_same_rm_singleA100.sh 42
```

Summarize evaluation metrics:

```bash
python -m benchmark.gpu.summarize_gpu_results \
  --input_glob 'outputs/gpu/**/eval*.json' \
  --output_csv outputs/gpu/results_summary.csv
```

## Multi-candidate DDO-RM and GRPO

DDO-RM K-sweep over K=2 plus generated candidate pools for K=4,8,16:

```bash
bash scripts/gpu/run_ddorm_k_sweep_singleA100.sh 42
```

To generate only the candidate pool for DDO-RM, run:

```bash
K=8 bash scripts/gpu/run_generate_candidates_singleA100.sh 42
```

GRPO with K=4 generations per prompt:

```bash
K=4 bash scripts/gpu/run_grpo_same_rm_singleA100.sh 42
```

GRPO with K=8 generations per prompt:

```bash
K=8 bash scripts/gpu/run_grpo_same_rm_singleA100.sh 42
```

## Full 3-seed run

```bash
bash scripts/gpu/run_full_gpu_benchmark_3seeds.sh
```

This runs seeds `42`, `13`, and `3407` for reward-model training, DDO-RM K=2, PPO, and GRPO K=4/K=8.

## Important metrics to report in the paper

For DDO-RM training, every run writes:

```text
outputs/gpu/<run_name>/train_metrics.jsonl
outputs/gpu/<run_name>/run_config.json
outputs/gpu/<run_name>/final_metrics.json
```

The most relevant DDO-RM diagnostics are:

```text
mean_kl_q_p
mean_target_entropy
mean_policy_entropy
mean_reward_gain_under_rm
eta_over_temperature
k
loss
examples_per_second_per_process
```

For evaluation, the scripts write:

```text
eval_pairwise_test_prefs.json
eval_listwise_k<K>.json
```

Pairwise metrics:

```text
pair_accuracy
auc
mean_margin
```

Listwise metrics:

```text
top1_accuracy_vs_gold
ndcg
pairwise_accuracy_vs_gold
spearman_vs_gold
mean_best_second_policy_margin
```

## Ablation knobs

DDO-RM:

```bash
ETA=0.3 TAU=1.0 bash scripts/gpu/run_ddorm_k2_single4090.sh 42
ETA=1.0 TAU=1.0 bash scripts/gpu/run_ddorm_k2_single4090.sh 42
ETA=3.0 TAU=1.0 bash scripts/gpu/run_ddorm_k2_single4090.sh 42
```

Reward robustness:

```bash
accelerate launch src/benchmark/gpu/train_ddorm_gpu.py \
  --model_name_or_path EleutherAI/pythia-410m \
  --reward_model_name_or_path outputs/gpu/reward_pythia410m_seed42 \
  --output_dir outputs/gpu/ddorm_noise1_seed42 \
  --dataset_kind uf_binarized \
  --split train_prefs \
  --seed 42 \
  --eta 1.0 \
  --decision_temperature 1.0 \
  --reward_noise_std 1.0 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-6 \
  --max_length 1024 \
  --dtype bf16 \
  --gradient_checkpointing \
  --use_lora
```

Distillation-step ablation:

```bash
accelerate launch src/benchmark/gpu/train_ddorm_gpu.py \
  --model_name_or_path EleutherAI/pythia-410m \
  --reward_model_name_or_path outputs/gpu/reward_pythia410m_seed42 \
  --output_dir outputs/gpu/ddorm_distill4_seed42 \
  --dataset_kind uf_binarized \
  --split train_prefs \
  --seed 42 \
  --eta 1.0 \
  --decision_temperature 1.0 \
  --distill_steps_per_batch 4 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-6 \
  --max_length 1024 \
  --dtype bf16 \
  --gradient_checkpointing \
  --use_lora
```

## Notes

The PPO script uses `trl.experimental.ppo.PPOTrainer`, whose API is newer than older TRL PPO examples. The GRPO script uses `trl.GRPOTrainer`. Keep `trl>=1.0.0` in the GPU environment.

The default scripts use LoRA to keep memory requirements manageable. If you want full fine-tuning, remove `--use_lora`, lower the batch size, and use DeepSpeed ZeRO as needed.

Do not commit checkpoints, optimizer states, or Hugging Face cache files. Commit only source files, configs, lightweight logs, JSON metrics, and summary CSVs.
