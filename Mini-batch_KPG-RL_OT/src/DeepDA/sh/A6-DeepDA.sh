#!/bin/bash
# =============================================================================
# A6 -- Number of keypoint pairs k, grid straddling the class count
#       (supervisor feedback #3, 2026-08-04)
# =============================================================================
# Grids per ablation-plan.md (class count in the INTERIOR of each grid):
#   Digits (10 cls):      k in {10, 20, 30, 40, 50}   via kp_per_class 1..5
#                         (k=5 needs a kp_n_classes flag digits/cfg.py lacks -- TODO)
#   VisDA (12 cls):       k in {6, 12, 24, 36, 48, 60} via kp_n_classes 6, then kp_per_class 1..5
#   Office-Home (65 cls): k in {15, 30, 65, 130, 195, 260} via kp_n_classes 15/30, then kp_per_class 1..4
#
# m is HELD FIXED across each sweep, large enough for the largest k (plan):
# Digits m=512, VisDA m=128, Office-Home m=512 -- a deliberate departure from
# the operating points, to be stated in the caption.
#
# m-KUOT SETTING ONLY, kp_strategy=random -- the plan notes that for
# kp_per_class > 1 the centroid/farthest strategies collapse to k = #classes
# (deterministic per-class picks collide), so only 'random' is valid here.
# k=0 is the plain m-UOT run (the code raises on zero keypoint pairs by
# design); it is gated behind RUN_BASELINE=1.
#
# Usage:  bash A6-DeepDA.sh [GPU]        (use a comma list, e.g. 0,1,2 -- the
#         fixed sweep batch sizes m=512/128 OOM a single 16 GB GPU at 224x224)
# RESUME: office runs skip via final-log grep; digits runs skip when
# snapshot/<desc>/final_model.pth exists (pc tag in the name for kp_per_class>1).
# Cost:   (5 x 3) Digits + 6 VisDA + (6 x 3) OH = 39 runs (+ baselines).
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ID="${RUN_ID:-0}"
ALPHA="${ALPHA:-0.5}"       # placeholder until A1 reports the argmax (plan note)
RHO="${RHO:-0.1}"
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

run_office_k () {  # run_office_k <dset> <s> <t> <task> <m> <eta1> <eta2> <tau> <per_class> <n_classes|-> <final_log>
  local DSET=$1 SPATH=$2 TPATH=$3 TASK=$4 M=$5 ETA1=$6 ETA2=$7 TAU=$8 PC=$9 NC=${10} FINAL_LOG=${11}
  local NC_FLAGS=() KTAG
  if [ "$NC" != "-" ]; then NC_FLAGS=(--kp_n_classes "${NC}"); KTAG="nc${NC}_pc${PC}"; else KTAG="pc${PC}"; fi
  local OUTPUT_DIR="A6_${DSET}_${TASK}_mKUOT_m${M}_${KTAG}_run${RUN_ID}"
  if [ -f "${FINAL_LOG}" ] && grep -q "snapshot/${OUTPUT_DIR}," "${FINAL_LOG}"; then
      echo "-- A6 ${TASK} ${KTAG}: already complete, skipping"; return
  fi
  echo "== A6 ${DSET} ${TASK}  m=${M}  ${KTAG} =="
  python train.py \
      --gpu_id "${GPU}" --net ResNet50 --dset "${DSET}" \
      --s_dset_path "${SPATH}" --t_dset_path "${TPATH}" \
      --stratify_source --batch_size "${M}" --test_interval 500 --stop_step 10000 \
      --test_10crop "${TEST_10CROP}" \
      --output_dir "${OUTPUT_DIR}" --final_log "${FINAL_LOG}" \
      --ot_type unbalanced --eta1 "${ETA1}" --eta2 "${ETA2}" --epsilon 0.01 --tau "${TAU}" --k 1 \
      --use_kpg --alpha "${ALPHA}" --kp_strategy random \
      --kp_per_class "${PC}" ${NC_FLAGS[@]+"${NC_FLAGS[@]}"} --rho "${RHO}"
}

# ---------------------------------------------------------------- Office-Home (m=512 fixed)
if [ "$RUN_OH" = "1" ]; then
cd "${SCRIPT_DIR}/../office"
DATA_ROOT="$(cd "${SCRIPT_DIR}/../../../data" && pwd)"
export OFFICE_HOME_IMAGES_ROOT="${DATA_ROOT}/office-home/images"
LIST_DIR="${DATA_ROOT}/office-home"
mkdir -p results
# HARDWARE LIMIT (2026-08-19): the plan's fixed m=512 cannot run here -- m>=256
# OOMs a 16 GB card even with AMP, and DataParallel was measured 80x slower
# (27 s/it vs 0.34 s/it single-GPU).  Capped at m=128, which forces k < 128, so
# the Office-Home grid covers k = 15, 30, 65 (up to one pair per class) and
# CANNOT straddle the class count from above.  VisDA (12 classes, k = 6..60 at
# m=128) is the benchmark that answers feedback #3's straddle requirement.
M=128
FINAL_LOG="results/A6_home_run${RUN_ID}_log.txt"
PAIRS=( "Art.txt Clipart.txt A2C" "Clipart.txt Art.txt C2A" "Product.txt Real_World.txt P2R" )
# k grid: (kp_n_classes, kp_per_class) -> k = 15, 30, 65   (130/195/260 need m>260)
K_GRID=( "15 1" "30 1" "- 1" )
for KENTRY in "${K_GRID[@]}"; do
  read -r NC PC <<< "${KENTRY}"
  for ENTRY in "${PAIRS[@]}"; do
    read -r SRC_FILE TGT_FILE TASK <<< "${ENTRY}"
    run_office_k office-home "${LIST_DIR}/${SRC_FILE}" "${LIST_DIR}/${TGT_FILE}" "${TASK}" ${M} 0.01 0.5 0.5 "${PC}" "${NC}" "${FINAL_LOG}"
  done
done
fi

# ---------------------------------------------------------------- VisDA-2017 (m=128 fixed)
if [ "$RUN_VISDA" = "1" ]; then
cd "${SCRIPT_DIR}/../office"
DEFAULT_VISDA_ROOT="${SCRIPT_DIR}/../../../../Baselines/Mini-batch-OT/DeepDA/office/data/visda-2017"
VISDA_ROOT="${VISDA_ROOT:-$DEFAULT_VISDA_ROOT}"
if [ -d "${VISDA_ROOT}/train" ]; then
export VISDA_IMAGES_ROOT="$(cd "${VISDA_ROOT}" && pwd)"
mkdir -p results
M=128
FINAL_LOG="results/A6_visda_run${RUN_ID}_log.txt"
# k grid: 6 (nc=6), then 12/24/36/48/60 via per_class 1..5
K_GRID=( "6 1" "- 1" "- 2" "- 3" "- 4" "- 5" )
for KENTRY in "${K_GRID[@]}"; do
  read -r NC PC <<< "${KENTRY}"
  run_office_k visda "${VISDA_IMAGES_ROOT}/train_list.txt" "${VISDA_IMAGES_ROOT}/validation_list.txt" T2V ${M} 0.005 1 0.3 "${PC}" "${NC}" "${FINAL_LOG}"
done
else echo "WARN: VisDA images not found under ${VISDA_ROOT}; skipping VisDA block"; fi
fi

# ---------------------------------------------------------------- Digits (m=512 fixed)
if [ "$RUN_DIGITS" = "1" ]; then
cd "${SCRIPT_DIR}/../digits"
DATA_ROOT="$(cd "${SCRIPT_DIR}/../../../data" && pwd)"
M=512
# TODO(plan): k=5 requires a --kp_n_classes flag in digits/cfg.py (not present).
for PC in 1 2 3 4 5; do   # k = 10, 20, 30, 40, 50
  for ENTRY in "svhn mnist" "usps mnist" "mnist usps"; do
    read -r SRC TGT <<< "${ENTRY}"
    PCTAG=""; [ "${PC}" != "1" ] && PCTAG="_pc${PC}"
    DESC="jumbot_${SRC}_to_${TGT}_k1_m${M}_lr0.0004_epsilon0.1_be0.0_mass0.85_tau1.0_kpg_a${ALPHA}_random${PCTAG}"
    if [ -f "snapshot/${DESC}/final_model.pth" ]; then
        echo "-- A6 Digits ${SRC}->${TGT} pc=${PC}: already complete, skipping"; continue
    fi
    echo "== A6 Digits ${SRC}->${TGT}  kp_per_class=${PC} (k=$((PC*10))) =="
    python train_digits.py \
        --gpu_id "${GPU}" --method jumbot --source_ds "${SRC}" --target_ds "${TGT}" \
        --k 1 --mbsize ${M} --n_epochs 100 --test_interval 1 --nclass 10 \
        --epsilon 0.1 --tau 1.0 --mass 0.85 --lr 4e-4 --eta1 0.1 --eta2 0.1 \
        --num_workers 8 --seed 1980 --data_dir "${DATA_ROOT}" \
        --use_kpg --alpha "${ALPHA}" --kp_strategy random --kp_per_class "${PC}" --rho "${RHO}"
  done
done
fi

# ---------------------------------------------------------------- k=0 baselines (m-UOT at the SAME fixed m)
if [ "$RUN_BASELINE" = "1" ]; then
  echo "== A6 k=0 baselines (m-UOT at the sweep's fixed m) =="
  cd "${SCRIPT_DIR}/../office"
  FINAL_LOG="results/A6_home_run${RUN_ID}_log.txt"
  DATA_ROOT="$(cd "${SCRIPT_DIR}/../../../data" && pwd)"
  export OFFICE_HOME_IMAGES_ROOT="${DATA_ROOT}/office-home/images"
  LIST_DIR="${DATA_ROOT}/office-home"
  PAIRS=( "Art.txt Clipart.txt A2C" "Clipart.txt Art.txt C2A" "Product.txt Real_World.txt P2R" )
  for ENTRY in "${PAIRS[@]}"; do
    read -r SRC_FILE TGT_FILE TASK <<< "${ENTRY}"
    OUTPUT_DIR="A6_office-home_${TASK}_mUOT_m128_k0_run${RUN_ID}"
    if [ -f "${FINAL_LOG}" ] && grep -q "snapshot/${OUTPUT_DIR}," "${FINAL_LOG}"; then continue; fi
    python train.py \
        --gpu_id "${GPU}" --net ResNet50 --dset office-home \
        --s_dset_path "${LIST_DIR}/${SRC_FILE}" --t_dset_path "${LIST_DIR}/${TGT_FILE}" \
        --stratify_source --batch_size 128 --test_interval 500 --stop_step 10000 \
        --test_10crop "${TEST_10CROP}" \
        --output_dir "${OUTPUT_DIR}" --final_log "${FINAL_LOG}" \
        --ot_type unbalanced --eta1 0.01 --eta2 0.5 --epsilon 0.01 --tau 0.5 --k 1
  done
fi
echo "A6-DeepDA done."
