#!/bin/bash
# =============================================================================
# A2 -- Batch size at fixed k, powers of two (supervisor feedback #2, 2026-08-04)
# =============================================================================
# m in {16, 32, 64, 128, 256, 512} subject to m > k (ablation-plan.md, A2):
#   Digits (k=10):      16 32 64 128 256 512   (512 = operating point)
#   VisDA (k=12):       16 32 64 128 256 512   (16 = m->k degenerate probe; 64 = op point)
#   Office-Home (k=65):          128 256 512   (128 = operating point; m<=64 infeasible)
#
# m-KUOT SETTING ONLY (ot_type=unbalanced + --use_kpg, kp_strategy=random).
# The matched-m m-UOT companion runs -- required by the plan to read the
# vertical gap as the effect of guidance -- are gated behind RUN_BASELINE=1.
# alpha is fixed at 0.5 (placeholder until A1 reports the argmax; plan note).
#
# Usage:  bash A2-DeepDA.sh [GPU]   (env: RUN_BASELINE=1 adds matched-m m-UOT)
#         bash A2-DeepDA.sh 0,1,2   (m >= 128 at 224x224 OOMs one 16 GB GPU;
#                                    train.py DataParallels over a comma list)
# RESUME: office runs skip via final-log grep; digits runs skip when
# snapshot/<desc>/final_model.pth exists.
# Cost:   m-KUOT only: 18 (Digits) + 6 (VisDA) + 9 (OH 3-pair) = 33 runs; x2 with baselines.
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ID="${RUN_ID:-0}"
ALPHA="${ALPHA:-0.5}"
RHO="${RHO:-0.1}"
KP_STRATEGY="${KP_STRATEGY:-random}"
RUN_BASELINE="${RUN_BASELINE:-0}"
# Evaluation protocol.  10-crop testing is ~73% of a run's wall time (measured on
# VisDA: 7.5 min per eval x 20 evals against 56 min of training).  Default True to
# stay consistent with the points already completed under it; pass
# TEST_10CROP=False for a curve whose points are ALL new.
TEST_10CROP="${TEST_10CROP:-True}"
# Mixed precision for the office trainer (halves activation memory: m=128
# fits ONE 16 GB GPU, m=256 two).  AMP=0 restores full fp32.
export MKUOT_AMP="${AMP:-1}"
export MKUOT_FAST_KP="${FAST_KP:-1}"   # cached keypoint injection (see train.py)
RUN_OH="${RUN_OH:-1}"; RUN_VISDA="${RUN_VISDA:-1}"; RUN_DIGITS="${RUN_DIGITS:-1}"

run_office () {  # run_office <dset> <s_path> <t_path> <task> <m> <eta1> <eta2> <tau> <kpg 0|1> <final_log>
  local DSET=$1 SPATH=$2 TPATH=$3 TASK=$4 M=$5 ETA1=$6 ETA2=$7 TAU=$8 KPG=$9 FINAL_LOG=${10}
  local METHOD=mUOT; [ "$KPG" = "1" ] && METHOD=mKUOT
  local OUTPUT_DIR="A2_${DSET}_${TASK}_${METHOD}_m${M}_alpha${ALPHA}_run${RUN_ID}"
  if [ -f "${FINAL_LOG}" ] && grep -q "snapshot/${OUTPUT_DIR}," "${FINAL_LOG}"; then
      echo "-- A2 ${TASK} ${METHOD} m=${M}: already complete, skipping"; return
  fi
  echo "== A2 ${DSET} ${TASK}  ${METHOD}  m=${M} =="
  local KPG_FLAGS=()
  [ "$KPG" = "1" ] && KPG_FLAGS=(--use_kpg --alpha "${ALPHA}" --kp_strategy "${KP_STRATEGY}" --kp_per_class 1 --rho "${RHO}")
  python train.py \
      --gpu_id "${GPU}" --net ResNet50 --dset "${DSET}" \
      --s_dset_path "${SPATH}" --t_dset_path "${TPATH}" \
      --stratify_source --batch_size "${M}" --test_interval 500 --stop_step 10000 \
      --test_10crop "${TEST_10CROP}" \
      --output_dir "${OUTPUT_DIR}" --final_log "${FINAL_LOG}" \
      --ot_type unbalanced --eta1 "${ETA1}" --eta2 "${ETA2}" --epsilon 0.01 --tau "${TAU}" --k 1 \
      ${KPG_FLAGS[@]+"${KPG_FLAGS[@]}"}
}

# ---------------------------------------------------------------- Office-Home
if [ "$RUN_OH" = "1" ]; then
cd "${SCRIPT_DIR}/../office"
DATA_ROOT="$(cd "${SCRIPT_DIR}/../../../data" && pwd)"
export OFFICE_HOME_IMAGES_ROOT="${DATA_ROOT}/office-home/images"
LIST_DIR="${DATA_ROOT}/office-home"
mkdir -p results
FINAL_LOG="results/A2_home_run${RUN_ID}_log.txt"
PAIRS=( "Art.txt Clipart.txt A2C" "Clipart.txt Art.txt C2A" "Product.txt Real_World.txt P2R" )
for M in 128 256 512; do
  for ENTRY in "${PAIRS[@]}"; do
    read -r SRC_FILE TGT_FILE TASK <<< "${ENTRY}"
    run_office office-home "${LIST_DIR}/${SRC_FILE}" "${LIST_DIR}/${TGT_FILE}" "${TASK}" "${M}" 0.01 0.5 0.5 1 "${FINAL_LOG}"
    [ "$RUN_BASELINE" = "1" ] && run_office office-home "${LIST_DIR}/${SRC_FILE}" "${LIST_DIR}/${TGT_FILE}" "${TASK}" "${M}" 0.01 0.5 0.5 0 "${FINAL_LOG}"
  done
done
fi

# ---------------------------------------------------------------- VisDA-2017
if [ "$RUN_VISDA" = "1" ]; then
cd "${SCRIPT_DIR}/../office"
DEFAULT_VISDA_ROOT="${SCRIPT_DIR}/../../../../Baselines/Mini-batch-OT/DeepDA/office/data/visda-2017"
VISDA_ROOT="${VISDA_ROOT:-$DEFAULT_VISDA_ROOT}"
if [ -d "${VISDA_ROOT}/train" ]; then
export VISDA_IMAGES_ROOT="$(cd "${VISDA_ROOT}" && pwd)"
mkdir -p results
FINAL_LOG="results/A2_visda_run${RUN_ID}_log.txt"
for M in 16 32 64 128 256 512; do   # m=16: 4 free slots, the m->k degenerate probe
    run_office visda "${VISDA_IMAGES_ROOT}/train_list.txt" "${VISDA_IMAGES_ROOT}/validation_list.txt" T2V "${M}" 0.005 1 0.3 1 "${FINAL_LOG}"
    [ "$RUN_BASELINE" = "1" ] && run_office visda "${VISDA_IMAGES_ROOT}/train_list.txt" "${VISDA_IMAGES_ROOT}/validation_list.txt" T2V "${M}" 0.005 1 0.3 0 "${FINAL_LOG}"
done
else echo "WARN: VisDA images not found under ${VISDA_ROOT}; skipping VisDA block"; fi
fi

# ---------------------------------------------------------------- Digits
if [ "$RUN_DIGITS" = "1" ]; then
cd "${SCRIPT_DIR}/../digits"
DATA_ROOT="$(cd "${SCRIPT_DIR}/../../../data" && pwd)"
for M in 16 32 64 128 256 512; do
  for ENTRY in "svhn mnist" "usps mnist" "mnist usps"; do
    read -r SRC TGT <<< "${ENTRY}"
    DESC="jumbot_${SRC}_to_${TGT}_k1_m${M}_lr0.0004_epsilon0.1_be0.0_mass0.85_tau1.0_kpg_a${ALPHA}_${KP_STRATEGY}"
    if [ -f "snapshot/${DESC}/final_model.pth" ]; then
        echo "-- A2 Digits ${SRC}->${TGT} m=${M}: already complete, skipping"
    else
    echo "== A2 Digits ${SRC}->${TGT}  m-KUOT  m=${M} =="
    python train_digits.py \
        --gpu_id "${GPU}" --method jumbot --source_ds "${SRC}" --target_ds "${TGT}" \
        --k 1 --mbsize "${M}" --n_epochs 100 --test_interval 1 --nclass 10 \
        --epsilon 0.1 --tau 1.0 --mass 0.85 --lr 4e-4 --eta1 0.1 --eta2 0.1 \
        --num_workers 8 --seed 1980 --data_dir "${DATA_ROOT}" \
        --use_kpg --alpha "${ALPHA}" --kp_strategy "${KP_STRATEGY}" --rho "${RHO}"
    fi
    if [ "$RUN_BASELINE" = "1" ] && [ ! -f "snapshot/jumbot_${SRC}_to_${TGT}_k1_m${M}_lr0.0004_epsilon0.1_be0.0_mass0.85_tau1.0/final_model.pth" ]; then
      echo "== A2 Digits ${SRC}->${TGT}  m-UOT  m=${M} =="
      python train_digits.py \
          --gpu_id "${GPU}" --method jumbot --source_ds "${SRC}" --target_ds "${TGT}" \
          --k 1 --mbsize "${M}" --n_epochs 100 --test_interval 1 --nclass 10 \
          --epsilon 0.1 --tau 1.0 --mass 0.85 --lr 4e-4 --eta1 0.1 --eta2 0.1 \
          --num_workers 8 --seed 1980 --data_dir "${DATA_ROOT}"
    fi
  done
done
fi
echo "A2-DeepDA done."
