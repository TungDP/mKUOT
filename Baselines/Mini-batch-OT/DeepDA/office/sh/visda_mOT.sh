#!/bin/bash
# VisDA-2017  |  mini-batch OT  (Synthetic train -> Real validation).
#
# VisDA is a SINGLE transfer, but the dataset is heavily class-imbalanced, so
# the standard metric is the MEAN of the 12 PER-CLASS accuracies (not overall
# accuracy).  train_visda.py now reports per-class accuracy and writes a
# results table; this script runs the transfer and prints that table.
#
# Knobs (override via env or first arg):
#   $1 / GPU_ID  : GPU id passed to train_visda.py            (default 0)
#   OT_TYPE      : balanced | unbalanced | partial            (default balanced)
#   USE_BOMB     : yes | no                                   (default no)
#   USE_MS       : 1 | 0  (Mirror Sinkhorn on/off)            (default 1)
#   MS_ITER      : Mirror Sinkhorn inner iterations           (default 500)
#   RUN_ID       : tag for parallel sweeps                    (default 0)
#   MASS         : POT transport mass                         (default 0.75)
#
# Usage:
#   bash sh/visda_mOT.sh 0
#   OT_TYPE=partial bash sh/visda_mOT.sh 0
#   USE_MS=0 bash sh/visda_mOT.sh 0          # baseline solver (exact EMD / entropic)
#
# Output:
#   home_visda_${TAG_LT}_log.txt   — train.py one-liner + per-class results table.
#   result_visda_table.md          — CUMULATIVE paper-style table (one row per run),
#       columns:  Run | Method | All | plane ... truck | Avg
#       ("All" = overall acc on 12 classes; "Avg" = mean per-class precision)
#   logs/visda/${TAG_LT}.log       — full training stdout.

set -uo pipefail

# ---------- knobs ----------------------------------------------------
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
OT_TYPE="${OT_TYPE:-balanced}"
USE_BOMB="${USE_BOMB:-no}"
USE_MS="${USE_MS:-1}"
MS_ITER="${MS_ITER:-500}"
RUN_ID="${RUN_ID:-0}"
MASS="${MASS:-0.75}"

# ---------- hyperparameters (VisDA defaults, cf. train_visda.sh) -----
ETA1=0.005
ETA2=1
TAU=0.3
ITER=10000
TEST_INTERVAL=500
M=72                # 12 classes x 6 per class (balanced sampler)
K=1
BATCH=$(( K * M ))
if [ "$OT_TYPE" = "unbalanced" ]; then
    EPSILON=0.01
else
    EPSILON=0
fi

# ---------- loss-type tag + optional flags ---------------------------
case "$OT_TYPE" in
    balanced)   LT=OT ;;
    unbalanced) LT=UOT ;;
    partial)    LT=POT ;;
    *) echo "Unknown OT_TYPE: $OT_TYPE" >&2; exit 1 ;;
esac
if [ "$USE_BOMB" = "yes" ]; then
    LT="BoMb_${LT}"
    USE_BOMB_FLAG="--use_bomb"
else
    USE_BOMB_FLAG=""
fi
if [ "$USE_MS" = "1" ]; then
    LT="${LT}_MS"
    USE_MS_FLAG="--use_mirror_sinkhorn --ms_iter ${MS_ITER}"
else
    USE_MS_FLAG=""
fi
TAG_LT="${LT}_run${RUN_ID}"

# ---------- output paths ---------------------------------------------
FINAL_LOG="home_visda_${TAG_LT}_log.txt"
LOG_DIR="logs/visda"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${TAG_LT}.log"
: > "$FINAL_LOG"            # fresh sweep — clear the summary/table file

S_PATH="./data/visda-2017/train_list.txt"
T_PATH="./data/visda-2017/validation_list.txt"
OUTPUT_DIR="visda_train_val_${TAG_LT}_mass${MASS}_k${K}_m${M}_eps${EPSILON}"

echo "===================================================================="
echo " VisDA-2017 mini-batch OT  (Synthetic -> Real)"
echo "   OT_TYPE=$OT_TYPE  USE_BOMB=$USE_BOMB  USE_MS=$USE_MS  MS_ITER=$MS_ITER"
echo "   K=$K  M=$M  BATCH=$BATCH  ITER=$ITER  MASS=$MASS  EPSILON=$EPSILON"
echo "   metric = mean per-class accuracy (12 classes)"
echo "   FINAL_LOG=$FINAL_LOG"
echo "===================================================================="

python -u train_visda.py --gpu_id "${GPU_ID}" \
                --net ResNet50 \
                --dset visda \
                --test_interval $TEST_INTERVAL \
                --s_dset_path "${S_PATH}" \
                --stratify_source \
                --t_dset_path "${T_PATH}" \
                --batch_size $BATCH \
                --output_dir "${OUTPUT_DIR}" \
                --final_log "${FINAL_LOG}" \
                --stop_step $ITER \
                --ot_type ${OT_TYPE} \
                --eta1 $ETA1 \
                --eta2 $ETA2 \
                --epsilon $EPSILON \
                --tau $TAU \
                --mass $MASS \
                --k $K \
                ${USE_BOMB_FLAG} \
                ${USE_MS_FLAG} 2>&1 | tee "$LOG_FILE"

echo
echo "=== VisDA-2017 per-class results table (best checkpoint) ==="
# train_visda.py appends a 3-line markdown table (header, sep, row) at the end.
tail -n 4 "$FINAL_LOG" 2>/dev/null | grep -E '^\|' || cat "$FINAL_LOG"

# ---------- append a row to the CUMULATIVE paper-style table ----------
RESULT_TABLE="result_visda_table.md"
VISDA_HEADER="| Run | Method | All | plane | bcycl | bus | car | horse | knife | mcycl | person | plant | sktbrd | train | truck | Avg |"
VISDA_SEP="|-----|--------|-----|-------|-------|-----|-----|-------|-------|-------|--------|-------|--------|-------|-------|-----|"
if [ ! -f "$RESULT_TABLE" ]; then
    { echo "$VISDA_HEADER"; echo "$VISDA_SEP"; } > "$RESULT_TABLE"
fi
# Parse the RESULT line train_visda.py wrote (All=.. plane=.. ... Avg=..) into
# pipe-separated values in column order, then prepend Run | Method.
RESULT_LINE=$(grep '^RESULT ' "$FINAL_LOG" | tail -n 1)
if [ -n "$RESULT_LINE" ]; then
    VALS=$(echo "$RESULT_LINE" | sed 's/^RESULT //' \
        | awk '{out=""; for(i=1;i<=NF;i++){split($i,a,"="); out=out (i>1?" | ":"") a[2]} print out}')
    echo "| ${RUN_ID} | ${LT} | ${VALS} |" >> "$RESULT_TABLE"
    echo
    echo "=== appended to cumulative ${RESULT_TABLE} ==="
    cat "$RESULT_TABLE"
else
    echo "WARNING: no RESULT line found in ${FINAL_LOG} (run may have crashed before stop_step)."
fi
echo
echo "Training Finished!!!  Per-run table in $FINAL_LOG ; cumulative table in $RESULT_TABLE."
