#!/bin/bash
# =============================================================================
# OH130 -- Office-Home main table re-run at m = 130 (decision 2026-08-25)
# =============================================================================
# WHY THIS EXISTS.  The Office-Home keypoint-guided numbers currently in the
# paper (Table `tab:officehome`, and the Office-Home panel of `fig:alpha`) were
# produced at m = 65 with k = 65, i.e. the DEGENERATE k == m configuration that
# `check_batch_feasibility` (office/train.py:67) raises on:
#
#   * the mask of Eq. (6) collapses to the identity, so the plan is forced
#     diagonal and no free transport remains;
#   * Eq. (9) defines g_{i,j} only for i,j > k, so with k == m the guiding
#     matrix is IDENTICALLY ZERO;
#   * hence C~ = alpha*Cbar + (1-alpha)*0 = alpha*Cbar -- alpha blends nothing
#     and only rescales the entropic temperature, eps -> eps/alpha.
#
# Those runs therefore do not evaluate the estimator of Section IV.  This script
# re-runs them at the m = 130 that Table `tab:hparams` already documents: 65
# reserved keypoint slots and 65 freely sampled ones, so k < m holds and both
# the mask and the guiding matrix are non-trivial.  See ablation-plan.md, R10.
#
# GRID.  12 transfer pairs x 3 target-keypoint strategies x ALPHAS.
#   ALPHAS="0.5"                 -> 36 runs, the operating point of the tables
#   ALPHAS="0.5 0.6 0.7 0.8 0.9" -> 180 runs, the full `fig:alpha` panel
# Everything else follows the MAIN-TABLE protocol, not the ablation protocol:
# 10-crop evaluation, 10 000 iterations, stratified source sampler (130/65 = 2
# source samples per class, as Table `tab:hparams` states).
#
# Usage:  bash OH130-DeepDA.sh [GPU]
#   # three GPUs, four pairs each (recommended -- ~1.7x faster than DataParallel)
#   OH_TAGS="A2C A2P A2R C2A" bash OH130-DeepDA.sh 0
#   OH_TAGS="C2P C2R P2A P2C" bash OH130-DeepDA.sh 1
#   OH_TAGS="P2R R2A R2C R2P" bash OH130-DeepDA.sh 2
#   # one strategy at a time
#   KP_STRATEGIES="random" OH_TAGS="A2C" bash OH130-DeepDA.sh 0
#   # matched-m m-UOT control (12 runs, no keypoints) -- required for the within-regime claim
#   RUN_BASELINE=1 OH_TAGS="A2C A2P A2R C2A" bash OH130-DeepDA.sh 0
#
# RESUME: a run is skipped when its line is already in the final log.  Atomic
# mkdir locks keep two invocations off the same config (this bit us on
# 2026-08-17 -- two writers interleaved into one CSV).
# COST: measured ~2 h/run at m=130 with 10-crop eval and AMP; 36 runs on three
# GPUs is roughly one day.
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ID="${RUN_ID:-0}"
RHO="${RHO:-0.1}"
ITER="${ITER:-10000}"
M="${M:-130}"                       # 65 keypoint slots + 65 sampled; k < m
TEST_10CROP="${TEST_10CROP:-True}"  # main-table protocol
# Mixed precision + cached keypoint injection: without these, m=130 does not fit
# a 16 GB card and kp_bank.load_batch decodes 2m JPEGs single-threaded per
# iteration (measured 4.5 s of a 5 s step).  AMP=0 / FAST_KP=0 disable.
export MKUOT_AMP="${AMP:-1}"
export MKUOT_FAST_KP="${FAST_KP:-1}"

read -ra ALPHAS <<< "${ALPHAS:-0.5}"
read -ra STRATEGIES <<< "${KP_STRATEGIES:-centroid random farthest}"
# RUN_BASELINE=1 runs the matched-m m-UOT control instead of the guided arms: same m, same
# protocol, NO keypoints anywhere (no --use_kpg, so no Eq. (5) injection, no mask, no guiding
# cost).  Table `tab:officehome` currently compares guided rows against an m-UOT row measured
# at m=65; once the guided rows move to m=130 that comparison is a protocol mismatch -- the
# very error this re-run exists to fix -- so the within-regime claim needs this control.
# alpha is irrelevant here and the runs are tagged without it.
RUN_BASELINE="${RUN_BASELINE:-0}"

# --- concurrency guard (atomic mkdir; stale PIDs reclaimed) -------------------
# LOCK_ROOT MUST be absolute.  It was ".ablation_locks" (relative) until 2026-09-03, and
# because this script cd's to office/ further down, every later `mkdir "${LOCK_ROOT}/<key>"`
# failed with "no such file or directory" -- which claim() below could not distinguish from
# "another process holds it", so it reported success and the locks were INERT for the whole
# sweep.  Anchor to SCRIPT_DIR so the CWD cannot matter.
LOCK_ROOT="${LOCK_ROOT:-${SCRIPT_DIR}/.ablation_locks}"
case "${LOCK_ROOT}" in /*) ;; *) echo "ERROR: LOCK_ROOT must be absolute: ${LOCK_ROOT}" >&2; exit 1;; esac
mkdir -p "${LOCK_ROOT}" || { echo "ERROR: cannot create lock root ${LOCK_ROOT}" >&2; exit 1; }
CLAIMED=()
claim () {
    local d="${LOCK_ROOT}/$1"
    if mkdir "${d}" 2>/dev/null; then echo $$ > "${d}/pid"; CLAIMED+=("${d}"); return 0; fi
    # mkdir failed.  Only ONE reason is benign: the directory already exists, i.e. somebody
    # holds the lock.  Any other failure (missing parent, permissions, full disk) means the
    # guard is not working, and silently continuing is how a sweep gets two writers on one
    # config.  Fail loudly instead of assuming we own it.
    if [ ! -d "${d}" ]; then
        echo "FATAL: lock '${d}' could not be created and does not exist -- the concurrency" >&2
        echo "       guard is broken; refusing to run unprotected." >&2
        exit 1
    fi
    local owner; owner="$(cat "${d}/pid" 2>/dev/null || true)"
    if [ -n "${owner}" ] && kill -0 "${owner}" 2>/dev/null; then return 1; fi
    echo $$ > "${d}/pid"; CLAIMED+=("${d}"); return 0
}
release_all () { local d; for d in "${CLAIMED[@]:-}"; do [ -n "${d}" ] && rm -rf "${d}"; done; }
trap release_all EXIT INT TERM

cd "${SCRIPT_DIR}/../office" || { echo "ERROR: cannot enter ${SCRIPT_DIR}/../office" >&2; exit 1; }
# A failed `cd` under `set -u` without `-e` leaves DATA_ROOT EMPTY and the run
# then trains on "/office-home/Art.txt" -- this exact bug cost a DeepGM sweep on
# 2026-08-19.  Fail fast instead, and allow an explicit override.
DATA_ROOT="${DATA_ROOT:-$(cd "${SCRIPT_DIR}/../../../data" 2>/dev/null && pwd)}"
if [ -z "${DATA_ROOT}" ] || [ ! -f "${DATA_ROOT}/office-home/Art.txt" ]; then
    echo "ERROR: Office-Home lists not found (looked for '${DATA_ROOT}/office-home/Art.txt')." >&2
    echo "       Set DATA_ROOT to the directory containing office-home/." >&2
    exit 1
fi
export OFFICE_HOME_IMAGES_ROOT="${DATA_ROOT}/office-home/images"
LIST_DIR="${DATA_ROOT}/office-home"
mkdir -p results

# All 12 ordered pairs of the standard protocol: "<src list> <tgt list> <tag>".
ALL_PAIRS=(
  "Art.txt        Clipart.txt     A2C"
  "Art.txt        Product.txt     A2P"
  "Art.txt        Real_World.txt  A2R"
  "Clipart.txt    Art.txt         C2A"
  "Clipart.txt    Product.txt     C2P"
  "Clipart.txt    Real_World.txt  C2R"
  "Product.txt    Art.txt         P2A"
  "Product.txt    Clipart.txt     P2C"
  "Product.txt    Real_World.txt  P2R"
  "Real_World.txt Art.txt         R2A"
  "Real_World.txt Clipart.txt     R2C"
  "Real_World.txt Product.txt     R2P"
)
# OH_TAGS selects a subset by tag, which is how the work is split across GPUs.
if [ -n "${OH_TAGS:-}" ]; then
  PAIRS=()
  for WANT in ${OH_TAGS}; do
    for ENTRY in "${ALL_PAIRS[@]}"; do
      read -r _ _ TAG <<< "${ENTRY}"
      [ "${TAG}" = "${WANT}" ] && PAIRS+=("${ENTRY}")
    done
  done
  if [ "${#PAIRS[@]}" -eq 0 ]; then echo "ERROR: OH_TAGS='${OH_TAGS}' matched no pair." >&2; exit 1; fi
else
  PAIRS=( "${ALL_PAIRS[@]}" )
fi

echo "== OH130: ${#PAIRS[@]} pair(s) x ${#STRATEGIES[@]} strategy(ies) x ${#ALPHAS[@]} alpha(s)" \
     "= $(( ${#PAIRS[@]} * ${#STRATEGIES[@]} * ${#ALPHAS[@]} )) runs  [m=${M}, 10crop=${TEST_10CROP}, GPU ${GPU}]"

if [ "${RUN_BASELINE}" = "1" ]; then
  FINAL_LOG="results/OH130_home_mUOT_run${RUN_ID}_log.txt"
  for ENTRY in "${PAIRS[@]}"; do
    read -r SRC_FILE TGT_FILE TASK <<< "${ENTRY}"
    OUTPUT_DIR="OH130_home_${TASK}_mUOT_m${M}_run${RUN_ID}"
    if [ -f "${FINAL_LOG}" ] && grep -q "snapshot/${OUTPUT_DIR}," "${FINAL_LOG}"; then
        echo "-- ${TASK} m-UOT baseline: already complete, skipping"; continue
    fi
    if ! claim "oh130_${TASK}_baseline"; then
        echo "-- ${TASK} m-UOT baseline: IN FLIGHT elsewhere, skipping"; continue
    fi
    echo "== OH130 ${TASK}  m-UOT baseline  (m=${M}, no keypoints) =="
    python train.py \
        --gpu_id "${GPU}" --net ResNet50 --dset office-home \
        --s_dset_path "${LIST_DIR}/${SRC_FILE}" --t_dset_path "${LIST_DIR}/${TGT_FILE}" \
        --stratify_source --batch_size "${M}" --test_interval 500 --stop_step "${ITER}" \
        --test_10crop "${TEST_10CROP}" \
        --output_dir "${OUTPUT_DIR}" --final_log "${FINAL_LOG}" \
        --ot_type unbalanced --eta1 0.01 --eta2 0.5 --epsilon 0.01 --tau 0.5 --k 1
  done
  echo "OH130-DeepDA baseline done."
  exit 0
fi

for KP_STRATEGY in "${STRATEGIES[@]}"; do
  FINAL_LOG="results/OH130_home_mKUOT_${KP_STRATEGY}_run${RUN_ID}_log.txt"
  for ALPHA in "${ALPHAS[@]}"; do
    for ENTRY in "${PAIRS[@]}"; do
      read -r SRC_FILE TGT_FILE TASK <<< "${ENTRY}"
      OUTPUT_DIR="OH130_home_${TASK}_mKUOT_m${M}_alpha${ALPHA}_${KP_STRATEGY}_run${RUN_ID}"
      if [ -f "${FINAL_LOG}" ] && grep -q "snapshot/${OUTPUT_DIR}," "${FINAL_LOG}"; then
          echo "-- ${TASK} a=${ALPHA} ${KP_STRATEGY}: already complete, skipping"; continue
      fi
      if ! claim "oh130_${TASK}_a${ALPHA}_${KP_STRATEGY}"; then
          echo "-- ${TASK} a=${ALPHA} ${KP_STRATEGY}: IN FLIGHT elsewhere, skipping"; continue
      fi
      echo "== OH130 ${TASK}  alpha=${ALPHA}  ${KP_STRATEGY}  (m=${M}) =="
      python train.py \
          --gpu_id "${GPU}" --net ResNet50 --dset office-home \
          --s_dset_path "${LIST_DIR}/${SRC_FILE}" --t_dset_path "${LIST_DIR}/${TGT_FILE}" \
          --stratify_source --batch_size "${M}" --test_interval 500 --stop_step "${ITER}" \
          --test_10crop "${TEST_10CROP}" \
          --output_dir "${OUTPUT_DIR}" --final_log "${FINAL_LOG}" \
          --ot_type unbalanced --eta1 0.01 --eta2 0.5 --epsilon 0.01 --tau 0.5 --k 1 \
          --use_kpg --alpha "${ALPHA}" --kp_strategy "${KP_STRATEGY}" \
          --kp_per_class 1 --rho "${RHO}"
    done
  done
done
echo "OH130-DeepDA done."
