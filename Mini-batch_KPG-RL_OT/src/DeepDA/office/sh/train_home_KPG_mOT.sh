#!/bin/bash
# Office-Home  |  KPG-RL + mini-batch OT  (averaging scheme WITH KPG guidance)
#
# Usage:
#   cd Mini-batch_KPG-RL_OT/src/DeepDA/sh
#   bash train_home_KPG_mOT.sh
#
# Tasks: all 12 Office-Home transfers

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script lives in src/DeepDA/office/sh ; train.py is in the parent (office/).
cd "${SCRIPT_DIR}/.."

# ── Dataset paths ──────────────────────────────────────────────────────────
# data/ is at the repo root: src/DeepDA/office/sh -> ../../../../data
DATA_ROOT="$(cd "${SCRIPT_DIR}/../../../../data" && pwd)"
mkdir -p results   # --final_log writes here (cwd is now office/)
export OFFICE_HOME_IMAGES_ROOT="${DATA_ROOT}/office-home/images"
LIST_DIR="${DATA_ROOT}/office-home"

# ── Hyper-parameters ───────────────────────────────────────────────────────
GPU=${1:-0}        # GPU id (first positional arg; default 0)
OT_TYPE=unbalanced
NET=ResNet50
EPSILON=0.01       # unbalanced OT requires epsilon > 0 for Sinkhorn;
                   # 0.01 follows the m-UOT paper (Appendix D.1)
ETA1=0.01
ETA2=0.5
TAU=0.5
MASS=0.65       # transported mass s -- FIXED at 0.65 in every setting; only alpha varies
K=1
# Mini-batch size m.  Office-Home closed-set has 65 classes and we build one keypoint pair
# per class, so k = 65.  The Sec. IV formulation reserves those k slots in every mini-batch
# (Eq. 5) and the mask of Eq. (6) pins them to their partners, so m MUST exceed k -- at
# m = 65 we would get m - k = 0: every slot a keypoint, mask = identity, no free transport,
# and check_batch_feasibility() raises.  m = 130 gives k = 65 reserved + 65 free slots and,
# via the class-balanced sampler (m // 65), 2 source samples per class instead of 1.
M=130
BATCH=$(( K * M ))
ITER=10000
TEST_INTERVAL=500
RUN_ID=0

# KPG-RL-KP parameters  (see keypoint_guided_OT.py kpg_rl_kp)
ALPHA=0.9           # combination coeff: alpha * C_norm + (1 - alpha) * G_norm
RHO=0.1             # relation-profile temperature (Eq. 8): softmax scale rho*max(c)
TARGET_KEYPOINTS=farthest  # target keypoint strategy: centroid | random | farthest

METHOD="KPG_mOT"
FINAL_LOG="results/home_${METHOD}_${OT_TYPE}_${ALPHA}_run${RUN_ID}_${TARGET_KEYPOINTS}_log.txt"

echo "=== Office-Home  |  ${METHOD}  |  run ${RUN_ID} ==="
echo "    KPG params: alpha=${ALPHA}  rho=${RHO}"

TASK_LIST=(
    "Art.txt       Clipart.txt    A2C"
    "Art.txt       Product.txt    A2P"
    "Art.txt       Real_World.txt A2R"
    "Clipart.txt   Art.txt        C2A"
    "Clipart.txt   Product.txt    C2P"
    "Clipart.txt   Real_World.txt C2R"
    "Product.txt   Art.txt        P2A"
    "Product.txt   Clipart.txt    P2C"
    "Product.txt   Real_World.txt P2R"
    "Real_World.txt Art.txt       R2A"
    "Real_World.txt Clipart.txt   R2C"
    "Real_World.txt Product.txt   R2P"
)

for ENTRY in "${TASK_LIST[@]}"; do
    read -r SRC_FILE TGT_FILE TASK <<< "${ENTRY}"
    S_PATH="${LIST_DIR}/${SRC_FILE}"
    T_PATH="${LIST_DIR}/${TGT_FILE}"
    OUTPUT_DIR="home_${TASK}_${METHOD}_${OT_TYPE}_k${K}_m${M}_eps${EPSILON}_alpha${ALPHA}_run${RUN_ID}"

    # ── Resume support ──────────────────────────────────────────────────
    # Skip tasks that already wrote a completed entry to FINAL_LOG.
    # (train.py wipes the snapshot dir and retrains from iter 0, so an
    #  interrupted task with no final-log entry is simply re-run.)
    if [ -f "${FINAL_LOG}" ] && grep -q "snapshot/${OUTPUT_DIR}," "${FINAL_LOG}"; then
        echo ""
        echo "── ${TASK}  →  already complete, skipping"
        continue
    fi

    echo ""
    echo "── ${TASK}  →  ${OUTPUT_DIR}"
    python train.py \
        --gpu_id        ${GPU} \
        --net           ${NET} \
        --dset          office-home \
        --s_dset_path   "${S_PATH}" \
        --t_dset_path   "${T_PATH}" \
        --stratify_source \
        --batch_size    ${BATCH} \
        --test_interval ${TEST_INTERVAL} \
        --stop_step     ${ITER} \
        --output_dir    "${OUTPUT_DIR}" \
        --final_log     "${FINAL_LOG}" \
        --ot_type       ${OT_TYPE} \
        --eta1          ${ETA1} \
        --eta2          ${ETA2} \
        --epsilon       ${EPSILON} \
        --tau           ${TAU} \
        --mass          ${MASS} \
        --k             ${K} \
        --use_kpg \
        --alpha         ${ALPHA} \
        --kp_strategy     ${TARGET_KEYPOINTS} \
        --rho ${RHO}
    echo "── Done: ${TASK}"
done

echo ""
echo "=== All Office-Home ${METHOD} tasks finished ==="
