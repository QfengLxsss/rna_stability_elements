#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
SPLIT_SET="${SPLIT_SET:-random0}"
SPLIT_NAME="${SPLIT_NAME:-}"
OUT_PREFIX="${OUT_PREFIX:-data/processed/ramht_small_${SPLIT_SET}}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
PATIENCE="${PATIENCE:-6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH_5UTR="${MAX_LENGTH_5UTR:-256}"
MAX_LENGTH_CDS="${MAX_LENGTH_CDS:-1024}"
MAX_LENGTH_3UTR="${MAX_LENGTH_3UTR:-1024}"
CODON_LENGTH="${CODON_LENGTH:-342}"
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
PREDICTION_ROLES="${PREDICTION_ROLES:-test}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
LOG_DIR="${LOG_DIR:-logs/ramht}"

mkdir -p "${LOG_DIR}"

LOG_PATH="${LOG_DIR}/$(basename "${OUT_PREFIX}").log"
PID_PATH="${LOG_DIR}/$(basename "${OUT_PREFIX}").pid"

echo "[launch] GPU_ID=${GPU_ID}"
if [[ -n "${SPLIT_NAME}" ]]; then
  echo "[launch] SPLIT_NAME=${SPLIT_NAME}"
else
  echo "[launch] SPLIT_SET=${SPLIT_SET}"
fi
echo "[launch] OUT_PREFIX=${OUT_PREFIX}"
echo "[launch] BATCH_SIZE=${BATCH_SIZE}"
echo "[launch] MODEL_DIM=${MODEL_DIM}"
echo "[launch] LOG_PATH=${LOG_PATH}"

COMMAND=(python scripts/run_ramht_multitask.py
  --device cuda
  --out-prefix "${OUT_PREFIX}"
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
  --fusion-mode "${FUSION_MODE}")
COMMAND+=(--prediction-roles "${PREDICTION_ROLES}")

if [[ "${SEPARATE_CODON_STREAM}" == "1" ]]; then
  COMMAND+=(--separate-codon-stream)
fi
if [[ "${TASK_SPECIFIC_GATES}" == "1" ]]; then
  COMMAND+=(--task-specific-gates)
fi
if [[ "${MASK_PADDING_ATTENTION}" == "1" ]]; then
  COMMAND+=(--mask-padding-attention)
fi
if [[ "${ENGINEERED_OUTPUT_SKIP}" == "1" ]]; then
  COMMAND+=(--engineered-output-skip)
fi

if [[ -n "${SPLIT_NAME}" ]]; then
  COMMAND+=(--split-name "${SPLIT_NAME}")
else
  COMMAND+=(--split-set "${SPLIT_SET}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  nohup "${COMMAND[@]}" > "${LOG_PATH}" 2>&1 &

echo "$!" > "${PID_PATH}"
echo "[launch] pid=$(cat "${PID_PATH}")"
echo "[launch] tail -f ${LOG_PATH}"
