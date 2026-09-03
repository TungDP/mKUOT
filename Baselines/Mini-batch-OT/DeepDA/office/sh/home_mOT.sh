#!/bin/bash
# Run ALL 12 Office-Home transfers for mini-batch OT.
#
# Knobs (override via env or first arg):
#   $1 / GPU_ID    : GPU id passed to train.py                 (default 0)
#   OT_TYPE        : balanced | unbalanced | partial           (default balanced)
#   USE_BOMB       : yes | no                                  (default no)
#   USE_MS         : 1 | 0  (Mirror Sinkhorn on/off)           (default 1)
#   MS_ITER        : Mirror Sinkhorn inner iterations          (default 500)
#   RUN_ID         : tag for parallel sweeps                   (default 0)
#   MASS           : POT transport mass                        (default 0.65)
#   RESUME_FROM    : transfer tag to resume at (e.g. "C2P")
#
# Usage:
#   bash sh/home_mOT.sh 0,1,2
#   OT_TYPE=partial bash sh/home_mOT.sh 0,1,2
#   USE_MS=0 bash sh/home_mOT.sh 0,1,2           # baseline POT solver
#   RESUME_FROM=P2A bash sh/home_mOT.sh 0,1,2
#
# Output (one final-summary file per sweep, in train.py's native format):
#   home_${TAG_LT}_log.txt    e.g.
#     method snapshot/home_A2C_..., iter: 09999, precision: 0.52623
#     method snapshot/home_A2P_..., iter: 09999, precision: 0.78123
#     ...
# Plus per-transfer stdout (for debugging) at:
#   logs/office-home/${TAG_LT}/${PAIR}.log

set -uo pipefail

# ---------- knobs ----------------------------------------------------
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
OT_TYPE="${OT_TYPE:-balanced}"
USE_BOMB="${USE_BOMB:-no}"
USE_MS="${USE_MS:-1}"
MS_ITER="${MS_ITER:-500}"
RUN_ID="${RUN_ID:-0}"
MASS="${MASS:-0.65}"

# ---------- hyperparameters (paper defaults) -------------------------
ETA1=0.01
ETA2=0.5
TAU=0.5
ITER=10000
TEST_INTERVAL=500
M=65
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

# ---------- output paths --------------------------------------------
# Single result file (train.py's --final_log target).  Each completed
# transfer appends ONE line of the form:
#   method snapshot/<output_dir>, iter: NNNNN, precision: 0.XXXXX
FINAL_LOG="home_${TAG_LT}_log.txt"

# Per-transfer stdout (live progress, for debugging / inspection).
LOG_DIR="logs/office-home/${TAG_LT}"
mkdir -p "$LOG_DIR"

# ---------- resume support ------------------------------------------
RESUME_FROM="${RESUME_FROM:-}"
started=0
if [ -z "$RESUME_FROM" ]; then
    started=1
    # Fresh sweep: clear the result file so we don't accumulate from
    # earlier (possibly crashed) runs.
    : > "$FINAL_LOG"
fi

# ---------- 12 Office-Home transfers ---------------------------------
DOMAINS=(Art Clipart Product Real_World)
DOMAIN_ABBR=(A C P R)

echo "===================================================================="
echo " Office-Home mini-batch OT sweep"
echo "   OT_TYPE=$OT_TYPE  USE_BOMB=$USE_BOMB  USE_MS=$USE_MS  MS_ITER=$MS_ITER"
echo "   K=$K  M=$M  BATCH=$BATCH  ITER=$ITER  MASS=$MASS  EPSILON=$EPSILON"
echo "   FINAL_LOG=$FINAL_LOG"
echo "===================================================================="

for s in 0 1 2 3; do
    for t in 0 1 2 3; do
        if [ "$s" -eq "$t" ]; then
            continue
        fi
        S_NAME="${DOMAINS[$s]}"
        T_NAME="${DOMAINS[$t]}"
        PAIR="${DOMAIN_ABBR[$s]}2${DOMAIN_ABBR[$t]}"

        if [ "$started" -eq 0 ]; then
            if [ "$PAIR" = "$RESUME_FROM" ]; then
                started=1
            else
                echo "----- Skipping already-completed transfer: $S_NAME -> $T_NAME -----"
                continue
            fi
        fi

        s_dset_path="./data/office-home/${S_NAME}.txt"
        t_dset_path="./data/office-home/${T_NAME}.txt"
        OUTPUT_DIR="home_${PAIR}_${TAG_LT}_mass${MASS}_k${K}_m${M}_eps${EPSILON}"
        LOG_FILE="${LOG_DIR}/${PAIR}.log"

        echo
        echo "===== Office-Home (${OT_TYPE}): $S_NAME -> $T_NAME ====="
        python -u train.py --gpu_id "${GPU_ID}" \
                        --net ResNet50 \
                        --dset office-home \
                        --test_interval $TEST_INTERVAL \
                        --s_dset_path "${s_dset_path}" \
                        --stratify_source \
                        --t_dset_path "${t_dset_path}" \
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
    done
done

echo
echo "Training Finished!!!  See $FINAL_LOG for the summary."
