#!/bin/bash
# =============================================================================
# A1 -- Guidance weight alpha, full curve (supervisor feedback #1, 2026-08-04)
# =============================================================================
# Sweeps alpha over the FULL range 0 -> 1 so the paper can state whether one
# alpha is good everywhere or each dataset needs its own (ablation-plan.md, A1).
# Staged per the plan: 0.1 steps on the primary benchmark (Office-Home, 3-pair
# subset A2C/C2A/P2R), 0.2 steps on VisDA-2017 and Digits.
#
# m-KUOT SETTING ONLY: ot_type=unbalanced + --use_kpg, practical strategy
# (kp_strategy=random).  The mask-off curve of A1 needs a --no_mask flag that
# does not exist yet (~3 lines at office/train.py:529) and is NOT run here.
# alpha=1.0 is the mask-only endpoint; the m-UOT baseline endpoint (no mask,
# no guidance) is the plain m-UOT run from the main tables.
#
# Batch sizes follow the adopted power-of-two operating points (feedback #2):
# Office-Home m=128, VisDA m=64 (open question 3: 128 is the alternative),
# Digits m=512.
#
# Usage:  bash A1-DeepDA.sh [GPU]        (env overrides: RUN_OH/RUN_VISDA/RUN_DIGITS=0|1)
#         bash A1-DeepDA.sh 0,1,2        (Office-Home m=128 OOMs one 16 GB GPU;
#                                         train.py DataParallels over a comma list)
# RESUME: completed office runs are skipped via the final-log grep; completed
# digits runs are skipped when snapshot/<desc>/final_model.pth exists.
#
# THROUGHPUT (recommended): run the three OH pairs IN PARALLEL, one per GPU --
# ~1.7x faster than DataParallel-ing each run over all three cards, e.g.:
#   PAIRS="Art.txt Clipart.txt A2C"        bash A1-DeepDA.sh 0   # terminal 1
#   PAIRS="Clipart.txt Art.txt C2A"        bash A1-DeepDA.sh 1   # terminal 2
#   PAIRS="Product.txt Real_World.txt P2R" bash A1-DeepDA.sh 2   # terminal 3
# (m=128 single-GPU needs ~13.6 of 15.5 GB -- works only if the card is
#  otherwise EMPTY; if another process holds memory, fall back to 0,1,2.)
# Staging: ALPHAS="0.0 0.2 0.4 0.6 0.8 1.0" runs a coarse grid first; refine
# later with the remaining values (resume-skip keeps completed runs).
# EVAL COST: ablation runs default to --test_10crop False (single-crop eval;
# 10-crop costs ~44k forward passes per test point).  The alpha curve is
# internally consistent either way; TEST_10CROP=True matches the main tables.
# PROGRESS: a tqdm bar appears on stderr AFTER the keypoint probe (encodes
# 2 x 4096 images first, several minutes); per-run logs stream into
# snapshot/<run>/log.txt at every test interval.
# Cost:   33 (OH) + 6 (VisDA) + 18 (Digits) = 57 runs.
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ID="${RUN_ID:-0}"
KP_STRATEGY="${KP_STRATEGY:-random}"
RHO="${RHO:-0.1}"
RUN_OH="${RUN_OH:-1}"; RUN_VISDA="${RUN_VISDA:-1}"; RUN_DIGITS="${RUN_DIGITS:-1}"

TEST_10CROP="${TEST_10CROP:-False}"
# A1 mask-off arm (NO_MASK=1): keypoints and guiding matrix unchanged, Eq. (6)
# replaced by all-ones.  Tagged separately so its snapshots, final log and locks
# never collide with the mask-on curves.
NO_MASK="${NO_MASK:-0}"
if [ "${NO_MASK}" = "1" ]; then MASK_FLAG=(--no_mask); MTAG="_nomask"; else MASK_FLAG=(); MTAG=""; fi
# KP_N_CLASSES restricts keypoints to the first N classes (the k-matched control).
# It MUST appear in the tag, or such a run collides with the full-k run of the same
# (pair, alpha, strategy) and gets skipped as "already complete".
if [ -n "${KP_N_CLASSES:-}" ]; then MTAG="${MTAG}_nc${KP_N_CLASSES}"; fi
# Mixed precision for the office trainer (halves activation memory: m=128
# fits ONE 16 GB GPU, m=256 two).  AMP=0 restores full fp32.
export MKUOT_AMP="${AMP:-1}"
export MKUOT_FAST_KP="${FAST_KP:-1}"   # cached keypoint injection (see train.py)
ITER="${ITER:-10000}"
read -ra ALPHAS_FINE <<< "${ALPHAS:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
read -ra ALPHAS_COARSE <<< "${ALPHAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
# --- concurrency guard -------------------------------------------------------
# A second invocation of this script (after a crash, or started on another GPU)
# would otherwise pick the same unfinished config as a live one and interleave
# writes into the same CSV / snapshot -- this actually happened to CIFAR
# alpha=1.0 on 2026-08-17.  `mkdir` is atomic, so it serves as the lock; locks
# whose owning PID is gone are reclaimed as stale.
# LOCK_ROOT MUST be absolute.  It was ".ablation_locks" (relative) until 2026-09-03, and
# because this script cd's to office/ further down, every later `mkdir "${LOCK_ROOT}/<key>"`
# failed with "no such file or directory" -- which claim() below could not distinguish from
# "another process holds it", so it reported success and the locks were INERT for the whole
# sweep.  Anchor to SCRIPT_DIR so the CWD cannot matter.
LOCK_ROOT="${LOCK_ROOT:-${SCRIPT_DIR}/.ablation_locks}"
case "${LOCK_ROOT}" in /*) ;; *) echo "ERROR: LOCK_ROOT must be absolute: ${LOCK_ROOT}" >&2; exit 1;; esac
mkdir -p "${LOCK_ROOT}" || { echo "ERROR: cannot create lock root ${LOCK_ROOT}" >&2; exit 1; }
CLAIMED=()
claim () {   # claim <key>  -> 0 if we now own it, 1 if a live process holds it
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


# ---------------------------------------------------------------- Office-Home
if [ "$RUN_OH" = "1" ]; then
cd "${SCRIPT_DIR}/../office"
DATA_ROOT="$(cd "${SCRIPT_DIR}/../../../data" && pwd)"
export OFFICE_HOME_IMAGES_ROOT="${DATA_ROOT}/office-home/images"
LIST_DIR="${DATA_ROOT}/office-home"
mkdir -p results
M=128            # adopted operating point (feedback #2); k=65 < m required
FINAL_LOG="results/A1_home_mKUOT_${KP_STRATEGY}${MTAG}_run${RUN_ID}_log.txt"
# Override with PAIRS="Art.txt Clipart.txt A2C" to run one pair on one GPU.
if [ -n "${PAIRS:-}" ]; then OH_PAIRS=( "${PAIRS}" ); else
  OH_PAIRS=( "Art.txt Clipart.txt A2C" "Clipart.txt Art.txt C2A" "Product.txt Real_World.txt P2R" )
fi
for ALPHA in "${ALPHAS_FINE[@]}"; do
  for ENTRY in "${OH_PAIRS[@]}"; do
    read -r SRC_FILE TGT_FILE TASK <<< "${ENTRY}"
    OUTPUT_DIR="A1_home_${TASK}_mKUOT_m${M}_alpha${ALPHA}_${KP_STRATEGY}${MTAG}_run${RUN_ID}"
    if [ -f "${FINAL_LOG}" ] && grep -q "snapshot/${OUTPUT_DIR}," "${FINAL_LOG}"; then
        echo "-- A1 OH ${TASK} alpha=${ALPHA}: already complete, skipping"; continue
    fi
    if ! claim "oh_${TASK}_a${ALPHA}_${KP_STRATEGY}${MTAG}"; then
        echo "-- A1 OH ${TASK} alpha=${ALPHA}: IN FLIGHT in another process, skipping"; continue
    fi
    echo "== A1 Office-Home ${TASK}  alpha=${ALPHA} =="
    python train.py \
        --gpu_id "${GPU}" --net ResNet50 --dset office-home \
        --s_dset_path "${LIST_DIR}/${SRC_FILE}" --t_dset_path "${LIST_DIR}/${TGT_FILE}" \
        --stratify_source --batch_size ${M} --test_interval 500 --stop_step "${ITER}" \
        --test_10crop "${TEST_10CROP}" \
        --output_dir "${OUTPUT_DIR}" --final_log "${FINAL_LOG}" \
        --ot_type unbalanced --eta1 0.01 --eta2 0.5 --epsilon 0.01 --tau 0.5 --k 1 \
        --use_kpg --alpha "${ALPHA}" --kp_strategy "${KP_STRATEGY}" \
        --kp_per_class "${KP_PER_CLASS:-1}" --rho "${RHO}" \
        ${KP_N_CLASSES:+--kp_n_classes "${KP_N_CLASSES}"} ${MASK_FLAG[@]+"${MASK_FLAG[@]}"}
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
M=64             # adopted power-of-two point (nearest to literature m=72)
FINAL_LOG="results/A1_visda_mKUOT_${KP_STRATEGY}${MTAG}_run${RUN_ID}_log.txt"
for ALPHA in "${ALPHAS_COARSE[@]}"; do
    OUTPUT_DIR="A1_visda_mKUOT_m${M}_alpha${ALPHA}_${KP_STRATEGY}${MTAG}_run${RUN_ID}"
    if [ -f "${FINAL_LOG}" ] && grep -q "snapshot/${OUTPUT_DIR}," "${FINAL_LOG}"; then
        echo "-- A1 VisDA alpha=${ALPHA}: already complete, skipping"; continue
    fi
    if ! claim "visda_a${ALPHA}_${KP_STRATEGY}${MTAG}"; then
        echo "-- A1 VisDA alpha=${ALPHA}: IN FLIGHT in another process, skipping"; continue
    fi
    echo "== A1 VisDA-2017  alpha=${ALPHA} =="
    python train.py \
        --gpu_id "${GPU}" --net ResNet50 --dset visda \
        --s_dset_path "${VISDA_IMAGES_ROOT}/train_list.txt" \
        --t_dset_path "${VISDA_IMAGES_ROOT}/validation_list.txt" \
        --stratify_source --batch_size ${M} --test_interval 500 --stop_step "${ITER}" \
        --test_10crop "${TEST_10CROP}" \
        --output_dir "${OUTPUT_DIR}" --final_log "${FINAL_LOG}" \
        --ot_type unbalanced --eta1 0.005 --eta2 1 --epsilon 0.01 --tau 0.3 --k 1 \
        --use_kpg --alpha "${ALPHA}" --kp_strategy "${KP_STRATEGY}" \
        --kp_per_class "${KP_PER_CLASS:-1}" --rho "${RHO}" \
        ${KP_N_CLASSES:+--kp_n_classes "${KP_N_CLASSES}"} ${MASK_FLAG[@]+"${MASK_FLAG[@]}"}
done
else echo "WARN: VisDA images not found under ${VISDA_ROOT}; skipping VisDA block"; fi
fi

# ---------------------------------------------------------------- Digits
if [ "$RUN_DIGITS" = "1" ]; then
cd "${SCRIPT_DIR}/../digits"
DATA_ROOT="$(cd "${SCRIPT_DIR}/../../../data" && pwd)"
M=512            # adopted power-of-two point (literature m=500)
for ALPHA in "${ALPHAS_COARSE[@]}"; do
  for ENTRY in "svhn mnist" "usps mnist" "mnist usps"; do
    read -r SRC TGT <<< "${ENTRY}"
    DESC="jumbot_${SRC}_to_${TGT}_k1_m${M}_lr0.0004_epsilon0.1_be0.0_mass0.85_tau1.0_kpg_a${ALPHA}_${KP_STRATEGY}"
    if [ -f "snapshot/${DESC}/final_model.pth" ]; then
        echo "-- A1 Digits ${SRC}->${TGT} alpha=${ALPHA}: already complete, skipping"; continue
    fi
    if ! claim "digits_${SRC}_${TGT}_a${ALPHA}_${KP_STRATEGY}"; then
        echo "-- A1 Digits ${SRC}->${TGT} alpha=${ALPHA}: IN FLIGHT in another process, skipping"; continue
    fi
    echo "== A1 Digits ${SRC}->${TGT}  alpha=${ALPHA} =="
    python train_digits.py \
        --gpu_id "${GPU}" --method jumbot --source_ds "${SRC}" --target_ds "${TGT}" \
        --k 1 --mbsize ${M} --n_epochs 100 --test_interval 1 --nclass 10 \
        --epsilon 0.1 --tau 1.0 --mass 0.85 --lr 4e-4 --eta1 0.1 --eta2 0.1 \
        --num_workers 8 --seed 1980 --data_dir "${DATA_ROOT}" \
        --use_kpg --alpha "${ALPHA}" --kp_strategy "${KP_STRATEGY}" --rho "${RHO}"
  done
done
fi
echo "A1-DeepDA done."
