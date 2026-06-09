#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SPLIT_SET="${SPLIT_SET:-all}"
RUN_ID="${RUN_ID:-ramht_2080ti_parallel}"
LOG_DIR="${LOG_DIR:-logs/ramht/${RUN_ID}}"
OUT_DIR="${OUT_DIR:-data/processed/ramht_runs/${RUN_ID}}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"

MAX_EPOCHS="${MAX_EPOCHS:-30}"
PATIENCE="${PATIENCE:-6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH_5UTR="${MAX_LENGTH_5UTR:-128}"
MAX_LENGTH_CDS="${MAX_LENGTH_CDS:-768}"
MAX_LENGTH_3UTR="${MAX_LENGTH_3UTR:-768}"
CODON_LENGTH="${CODON_LENGTH:-256}"
MODEL_DIM="${MODEL_DIM:-128}"
CODON_DIM="${CODON_DIM:-128}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-2}"
CODON_LAYERS="${CODON_LAYERS:-1}"
ATTENTION_HEADS="${ATTENTION_HEADS:-4}"
FEEDFORWARD_DIM="${FEEDFORWARD_DIM:-256}"
FEATURE_HIDDEN_DIM="${FEATURE_HIDDEN_DIM:-256}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-128}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
FUSION_MODE="${FUSION_MODE:-gated_sum}"
SEPARATE_CODON_STREAM="${SEPARATE_CODON_STREAM:-0}"
TASK_SPECIFIC_GATES="${TASK_SPECIFIC_GATES:-0}"
MASK_PADDING_ATTENTION="${MASK_PADDING_ATTENTION:-0}"
ENGINEERED_OUTPUT_SKIP="${ENGINEERED_OUTPUT_SKIP:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "No GPUs specified via GPUS." >&2
  exit 1
fi

mapfile -t SPLITS < <(
  PYTHONPATH=src python - <<'PY'
import os
import pandas as pd

split_set = os.environ.get("SPLIT_SET", "all")
manifest = pd.read_csv(
    "data/processed/fair_benchmark_splits_gene_sense_late_chase_6h_2h.tsv",
    sep="\t",
)
splits = manifest["split_name"].drop_duplicates().tolist()
if split_set == "random0":
    splits = ["random_repeat_0"]
elif split_set == "random":
    splits = [item for item in splits if item.startswith("random_repeat_")]
elif split_set == "chromosome":
    splits = [item for item in splits if item.startswith("holdout_")]
elif split_set != "all":
    requested = [item.strip() for item in split_set.split(",") if item.strip()]
    splits = requested
for split in splits:
    print(split)
PY
)

echo "[ramht-parallel] RUN_ID=${RUN_ID}"
echo "[ramht-parallel] GPUS=${GPUS}"
echo "[ramht-parallel] SPLIT_SET=${SPLIT_SET}"
echo "[ramht-parallel] n_splits=${#SPLITS[@]}"
echo "[ramht-parallel] OUT_DIR=${OUT_DIR}"
echo "[ramht-parallel] LOG_DIR=${LOG_DIR}"

max_jobs=$(( ${#GPU_ARRAY[@]} * MAX_JOBS_PER_GPU ))
launched=0

for split_name in "${SPLITS[@]}"; do
  while [[ "$(jobs -rp | wc -l)" -ge "${max_jobs}" ]]; do
    sleep 10
  done
  gpu="${GPU_ARRAY[$((launched % ${#GPU_ARRAY[@]}))]}"
  split_safe="${split_name//[^A-Za-z0-9_]/_}"
  out_prefix="${OUT_DIR}/${RUN_ID}_${split_safe}"
  log_path="${LOG_DIR}/${split_safe}.log"
  pid_path="${LOG_DIR}/${split_safe}.pid"
  if [[ -s "${out_prefix}_metrics.tsv" ]]; then
    echo "[skip] ${split_name}: ${out_prefix}_metrics.tsv exists"
    launched=$((launched + 1))
    continue
  fi
  echo "[start] split=${split_name} gpu=${gpu}"
  command=(
    python scripts/run_ramht_multitask.py
      --device cuda
      --split-name "${split_name}"
      --out-prefix "${out_prefix}"
      --max-epochs "${MAX_EPOCHS}"
      --patience "${PATIENCE}"
      --batch-size "${BATCH_SIZE}"
      --learning-rate "${LEARNING_RATE}"
      --max-length-5utr "${MAX_LENGTH_5UTR}"
      --max-length-cds "${MAX_LENGTH_CDS}"
      --max-length-3utr "${MAX_LENGTH_3UTR}"
      --codon-length "${CODON_LENGTH}"
      --model-dim "${MODEL_DIM}"
      --codon-dim "${CODON_DIM}"
      --transformer-layers "${TRANSFORMER_LAYERS}"
      --codon-layers "${CODON_LAYERS}"
      --attention-heads "${ATTENTION_HEADS}"
      --feedforward-dim "${FEEDFORWARD_DIM}"
      --feature-hidden-dim "${FEATURE_HIDDEN_DIM}"
      --head-hidden-dim "${HEAD_HIDDEN_DIM}"
      --fusion-mode "${FUSION_MODE}"
  )
  if [[ "${SEPARATE_CODON_STREAM}" == "1" ]]; then
    command+=(--separate-codon-stream)
  fi
  if [[ "${TASK_SPECIFIC_GATES}" == "1" ]]; then
    command+=(--task-specific-gates)
  fi
  if [[ "${MASK_PADDING_ATTENTION}" == "1" ]]; then
    command+=(--mask-padding-attention)
  fi
  if [[ "${ENGINEERED_OUTPUT_SKIP}" == "1" ]]; then
    command+=(--engineered-output-skip)
  fi
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH=src \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    "${command[@]}"
  ) > "${log_path}" 2>&1 &
  echo "$!" > "${pid_path}"
  launched=$((launched + 1))
done

status=0
for job in $(jobs -rp); do
  if ! wait "${job}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "[ramht-parallel] one or more splits failed; inspect ${LOG_DIR}" >&2
  exit "${status}"
fi

PYTHONPATH=src python scripts/summarize_ramht_runs.py \
  --run-dir "${OUT_DIR}" \
  --out-prefix "data/processed/${RUN_ID}"

echo "[ramht-parallel] done"
