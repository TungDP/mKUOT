#!/bin/bash
# =============================================================================
# Q1 -- t-SNE embedding visualisation (supervisor feedback #4, 2026-08-04)
# =============================================================================
# The qualitative figure of the m-POT paper (its Figs. 11/12), reproduced for
# the m-KUOT setting: three panels -- m-UOT (baseline) | m-KUOT-r (practical) |
# m-KUOT-c (oracle) -- samples coloured by class, source as circles, target as
# '+' (plot_embeddings.py, no code change needed for three panels).
#
# Stage 1 trains the three configurations WITH checkpoint retention (train.py
# saves best_model.pth on every best-accuracy improvement; the old snapshots
# predate that code, which is why these runs are needed -- ablation-plan.md, Q1).
# Stage 2 renders the panel.  Office-Home A2C (hardest pair, largest margin);
# optional VisDA block (RUN_VISDA=1) mirrors m-POT's Fig. 12.
#
# The m-UOT panel is the baseline column of the figure itself, so it is NOT
# gated here: without it the figure cannot make its comparison.
#
# Usage:  bash Q1-DeepDA.sh [GPU]      (env: RUN_VISDA=1 adds the VisDA figure)
# Cost:   3 training runs per benchmark + minutes for t-SNE.
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../office"
GPU="${1:-0}"
RUN_ID="${RUN_ID:-0}"
ALPHA="${ALPHA:-0.5}"       # placeholder until A1 reports the argmax
RHO="${RHO:-0.1}"
RUN_VISDA="${RUN_VISDA:-0}"
TEST_10CROP="${TEST_10CROP:-False}"          # single-crop eval (speed); True = main-table protocol
# Mixed precision + cached keypoint batching (see train.py): required for the
# m=128 Office-Home runs to fit a single 16 GB GPU.  AMP=0 / FAST_KP=0 disable.
export MKUOT_AMP="${AMP:-1}"
export MKUOT_FAST_KP="${FAST_KP:-1}"
DATA_ROOT="$(cd "${SCRIPT_DIR}/../../../data" && pwd)"
export OFFICE_HOME_IMAGES_ROOT="${DATA_ROOT}/office-home/images"
LIST_DIR="${DATA_ROOT}/office-home"
mkdir -p results
M=128
FINAL_LOG="results/Q1_home_run${RUN_ID}_log.txt"
S_PATH="${LIST_DIR}/Art.txt"; T_PATH="${LIST_DIR}/Clipart.txt"

train_one () {  # train_one <output_dir> <extra flags...>
  local OUTPUT_DIR=$1; shift
  if [ -f "snapshot/${OUTPUT_DIR}/best_model.pth" ]; then
      echo "-- ${OUTPUT_DIR}: checkpoint exists, skipping"; return
  fi
  python train.py \
      --gpu_id "${GPU}" --net ResNet50 --dset office-home \
      --s_dset_path "${S_PATH}" --t_dset_path "${T_PATH}" \
      --stratify_source --batch_size ${M} --test_interval 500 --stop_step 10000 \
      --test_10crop "${TEST_10CROP}" \
      --output_dir "${OUTPUT_DIR}" --final_log "${FINAL_LOG}" \
      --ot_type unbalanced --eta1 0.01 --eta2 0.5 --epsilon 0.01 --tau 0.5 --k 1 "$@"
}

# ---- Stage 1: the three configurations (checkpoints retained automatically)
DIR_UOT="Q1_home_A2C_mUOT_m${M}_run${RUN_ID}"
DIR_R="Q1_home_A2C_mKUOT-r_m${M}_alpha${ALPHA}_run${RUN_ID}"
DIR_C="Q1_home_A2C_mKUOT-c_m${M}_alpha${ALPHA}_run${RUN_ID}"
echo "== Q1 Office-Home A2C: m-UOT baseline ==";  train_one "${DIR_UOT}"
echo "== Q1 Office-Home A2C: m-KUOT-r ==";        train_one "${DIR_R}" --use_kpg --alpha "${ALPHA}" --kp_strategy random   --kp_per_class 1 --rho "${RHO}"
echo "== Q1 Office-Home A2C: m-KUOT-c (oracle) ==";train_one "${DIR_C}" --use_kpg --alpha "${ALPHA}" --kp_strategy centroid --kp_per_class 1 --rho "${RHO}"

# ---- Stage 2: t-SNE panel (ratio ~0.13 lands near m-POT's 2000 samples on OH;
#      the script's own config keeps test_10crop off -- do not enable it)
echo "== Q1 t-SNE panel (Office-Home A2C) =="
python plot_embeddings.py \
    --gpu_id "${GPU}" --net ResNet50 --dset office-home \
    --s_dset_path "${S_PATH}" --t_dset_path "${T_PATH}" \
    --restore_dir "${DIR_UOT}" "${DIR_R}" "${DIR_C}" \
    --titles "m-UOT" "m-KUOT-r" "m-KUOT-c" \
    --output_dir "Q1_home_A2C_tsne_run${RUN_ID}" \
    --ratio "${RATIO:-0.5}" --batch_size 100

# ---------------------------------------------------------------- optional VisDA
if [ "$RUN_VISDA" = "1" ]; then
DEFAULT_VISDA_ROOT="${SCRIPT_DIR}/../../../../Baselines/Mini-batch-OT/DeepDA/office/data/visda-2017"
VISDA_ROOT="${VISDA_ROOT:-$DEFAULT_VISDA_ROOT}"
if [ -d "${VISDA_ROOT}/train" ]; then
export VISDA_IMAGES_ROOT="$(cd "${VISDA_ROOT}" && pwd)"
S_PATH="${VISDA_IMAGES_ROOT}/train_list.txt"; T_PATH="${VISDA_IMAGES_ROOT}/validation_list.txt"
M=64
FINAL_LOG="results/Q1_visda_run${RUN_ID}_log.txt"
train_one_v () {
  local OUTPUT_DIR=$1; shift
  if [ -f "snapshot/${OUTPUT_DIR}/best_model.pth" ]; then
      echo "-- ${OUTPUT_DIR}: checkpoint exists, skipping"; return
  fi
  python train.py \
      --gpu_id "${GPU}" --net ResNet50 --dset visda \
      --s_dset_path "${S_PATH}" --t_dset_path "${T_PATH}" \
      --stratify_source --batch_size ${M} --test_interval 500 --stop_step 10000 \
      --test_10crop "${TEST_10CROP}" \
      --output_dir "${OUTPUT_DIR}" --final_log "${FINAL_LOG}" \
      --ot_type unbalanced --eta1 0.005 --eta2 1 --epsilon 0.01 --tau 0.3 --k 1 "$@"
}
DIR_UOT="Q1_visda_mUOT_m${M}_run${RUN_ID}"
DIR_R="Q1_visda_mKUOT-r_m${M}_alpha${ALPHA}_run${RUN_ID}"
DIR_C="Q1_visda_mKUOT-c_m${M}_alpha${ALPHA}_run${RUN_ID}"
echo "== Q1 VisDA: m-UOT baseline ==";   train_one_v "${DIR_UOT}"
echo "== Q1 VisDA: m-KUOT-r ==";         train_one_v "${DIR_R}" --use_kpg --alpha "${ALPHA}" --kp_strategy random   --kp_per_class 1 --rho "${RHO}"
echo "== Q1 VisDA: m-KUOT-c (oracle) =="; train_one_v "${DIR_C}" --use_kpg --alpha "${ALPHA}" --kp_strategy centroid --kp_per_class 1 --rho "${RHO}"
echo "== Q1 t-SNE panel (VisDA) =="
python plot_embeddings.py \
    --gpu_id "${GPU}" --net ResNet50 --dset visda \
    --s_dset_path "${S_PATH}" --t_dset_path "${T_PATH}" \
    --restore_dir "${DIR_UOT}" "${DIR_R}" "${DIR_C}" \
    --titles "m-UOT" "m-KUOT-r" "m-KUOT-c" \
    --output_dir "Q1_visda_tsne_run${RUN_ID}" \
    --ratio "${RATIO_VISDA:-0.036}" --batch_size 100
else echo "WARN: VisDA images not found under ${VISDA_ROOT}; skipping VisDA block"; fi
fi
echo "Q1-DeepDA done."
