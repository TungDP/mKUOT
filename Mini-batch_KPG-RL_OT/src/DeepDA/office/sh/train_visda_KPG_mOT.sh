#!/bin/bash
# VisDA-2017  |  m-KPOT  (mini-batch Keypoint-Guided OT, averaging scheme).
#
# VisDA is a SINGLE transfer (Synthetic train -> Real validation), but it is
# heavily class-imbalanced, so the metric is the MEAN of the 12 PER-CLASS
# accuracies.  train.py reports per-class accuracy and writes a results table:
#   All | plane ... truck | Avg     (All = overall accuracy on the 12 classes).
#
# Knobs (override via env or first arg):
#   $1 / GPU       : GPU id                                   (default 0)
#   OT_TYPE        : balanced | unbalanced | partial          (default partial)
#   KP_STRATEGY    : centroid | random | farthest             (default random)
#   ALPHA          : KPG blend  cost = ALPHA*C + (1-ALPHA)*G   (default 0.5)
#   RHO            : relation temperature of Eq. (8)           (default 0.1)
#   RUN_ID         : tag for parallel sweeps                  (default 0)
#   MASS           : POT transport mass s                     (default 0.65, fixed)
#   VISDA_ROOT     : dir holding train/ validation/ + *_list  (default: the
#                    baseline copy under Baselines/Mini-batch-OT/.../visda-2017)
#
# Usage:
#   bash sh/train_visda_KPG_mOT.sh 0
#   OT_TYPE=balanced KP_STRATEGY=centroid bash sh/train_visda_KPG_mOT.sh 0
#
# Output:
#   results/visda_KPG_${TAG}_log.txt  — train.py one-liner + per-class table
#   result_visda_KPG_table.md         — CUMULATIVE paper-style table (one row/run):
#       | Run | Method | All | plane ... truck | Avg |
#   logs/visda/${TAG}.log             — full training stdout.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."          # train.py lives in office/

# ---------- knobs ----------------------------------------------------
GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
OT_TYPE="${OT_TYPE:-partial}"
KP_STRATEGY="${KP_STRATEGY:-random}"
ALPHA="${ALPHA:-0.5}"
RUN_ID="${RUN_ID:-0}"
MASS="${MASS:-0.65}"       # transported mass s -- FIXED at 0.65 in every setting; only alpha varies

# ---------- VisDA data location --------------------------------------
# VisDA images + list files (the KPG repo has no local copy, so we reuse the
# baseline's download by default; override with VISDA_ROOT=...).
DEFAULT_VISDA_ROOT="${SCRIPT_DIR}/../../../../../Baselines/Mini-batch-OT/DeepDA/office/data/visda-2017"
VISDA_ROOT="${VISDA_ROOT:-$DEFAULT_VISDA_ROOT}"
if [ ! -d "${VISDA_ROOT}/train" ] || [ ! -d "${VISDA_ROOT}/validation" ]; then
    echo "ERROR: VisDA images not found under '${VISDA_ROOT}'." >&2
    echo "       Set VISDA_ROOT to a dir containing train/ validation/ + *_list.txt." >&2
    exit 1
fi
VISDA_ROOT="$(cd "${VISDA_ROOT}" && pwd)"
export VISDA_IMAGES_ROOT="${VISDA_ROOT}"        # train.py resolves image paths here
S_PATH="${VISDA_ROOT}/train_list.txt"
T_PATH="${VISDA_ROOT}/validation_list.txt"

# ---------- hyperparameters (VisDA defaults) -------------------------
NET=ResNet50
ETA1=0.005
ETA2=1
TAU=0.3
ITER=10000
TEST_INTERVAL=500
M=72              # 12 classes x 6 per class (balanced sampler)
K=1
BATCH=$(( K * M ))
RHO="${RHO:-0.1}"   # relation-profile temperature (Eq. 8): softmax scale rho*max(c)
if [ "$OT_TYPE" = "unbalanced" ]; then
    EPSILON=0.01  # unbalanced OT requires entropic reg for Sinkhorn
else
    EPSILON=0     # balanced / partial use exact LP
fi

# ---------- tags + output paths --------------------------------------
case "$OT_TYPE" in
    balanced)   LT=OT ;;
    unbalanced) LT=UOT ;;
    partial)    LT=POT ;;
    *) echo "Unknown OT_TYPE: $OT_TYPE" >&2; exit 1 ;;
esac
TAG="KPG_${LT}_${KP_STRATEGY}_a${ALPHA}_run${RUN_ID}"
METHOD="KPG_${LT}_${KP_STRATEGY}"

mkdir -p results logs/visda
FINAL_LOG="results/visda_${TAG}_log.txt"
LOG_FILE="logs/visda/${TAG}.log"
: > "$FINAL_LOG"
OUTPUT_DIR="visda_${TAG}_mass${MASS}_k${K}_m${M}_eps${EPSILON}"

echo "===================================================================="
echo " VisDA-2017 m-KPOT  (Synthetic -> Real)"
echo "   OT_TYPE=$OT_TYPE  KP_STRATEGY=$KP_STRATEGY  ALPHA=$ALPHA  RHO=$RHO"
echo "   K=$K  M=$M  BATCH=$BATCH  ITER=$ITER  MASS=$MASS  EPSILON=$EPSILON"
echo "   metric = mean per-class accuracy (12 classes)"
echo "   VISDA_ROOT=$VISDA_ROOT"
echo "   FINAL_LOG=$FINAL_LOG"
echo "===================================================================="

python -u train.py --gpu_id "${GPU}" \
    --net ${NET} \
    --dset visda \
    --test_interval ${TEST_INTERVAL} \
    --s_dset_path "${S_PATH}" \
    --stratify_source \
    --t_dset_path "${T_PATH}" \
    --batch_size ${BATCH} \
    --output_dir "${OUTPUT_DIR}" \
    --final_log "${FINAL_LOG}" \
    --stop_step ${ITER} \
    --ot_type ${OT_TYPE} \
    --eta1 ${ETA1} \
    --eta2 ${ETA2} \
    --epsilon ${EPSILON} \
    --tau ${TAU} \
    --mass ${MASS} \
    --k ${K} \
    --use_kpg \
    --alpha ${ALPHA} \
    --kp_strategy ${KP_STRATEGY} \
    --rho ${RHO}

echo
echo "=== VisDA-2017 per-class results table (best checkpoint) ==="
tail -n 4 "$FINAL_LOG" 2>/dev/null | grep -E '^\|' || cat "$FINAL_LOG"

# ---------- append a row to the CUMULATIVE paper-style table ----------
RESULT_TABLE="result_visda_KPG_table.md"
VISDA_HEADER="| Run | Method | All | plane | bcycl | bus | car | horse | knife | mcycl | person | plant | sktbrd | train | truck | Avg |"
VISDA_SEP="|-----|--------|-----|-------|-------|-----|-----|-------|-------|-------|--------|-------|--------|-------|-------|-----|"
if [ ! -f "$RESULT_TABLE" ]; then
    { echo "$VISDA_HEADER"; echo "$VISDA_SEP"; } > "$RESULT_TABLE"
fi
RESULT_LINE=$(grep '^RESULT ' "$FINAL_LOG" | tail -n 1)
if [ -n "$RESULT_LINE" ]; then
    VALS=$(echo "$RESULT_LINE" | sed 's/^RESULT //' \
        | awk '{out=""; for(i=1;i<=NF;i++){split($i,a,"="); out=out (i>1?" | ":"") a[2]} print out}')
    echo "| ${RUN_ID} | ${METHOD} | ${VALS} |" >> "$RESULT_TABLE"
    echo
    echo "=== appended to cumulative ${RESULT_TABLE} ==="
    cat "$RESULT_TABLE"
else
    echo "WARNING: no RESULT line in ${FINAL_LOG} (run may have crashed before stop_step)."
fi
echo
echo "Training Finished!!!  Per-run table in $FINAL_LOG ; cumulative table in $RESULT_TABLE."
