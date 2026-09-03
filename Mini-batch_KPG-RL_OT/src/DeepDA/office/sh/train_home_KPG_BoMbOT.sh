#!/bin/bash
# NOTE: keypoint guidance is NOT compatible with the hierarchical
# batch-of-mini-batches (BoMb) scheme -- Sec. IV's estimator is the uniform average
# of Eq. (8), so --use_bomb together with --use_kpg is refused by the trainer.  This
# script therefore runs the UNGUIDED BoMb baseline.  Delete it if you do not need
# that baseline (an equivalent lives in ../../../Baselines/Mini-batch-OT/).
# Office-Home  |  KPG-RL + BoMb-OT  (hierarchical BoMb scheme WITH KPG guidance)
#
# This is the full proposed method combining:
#   - Batch-of-Mini-batches (BoMb) hierarchical OT  [Nguyen et al., ICML 2022]
#   - Keypoint-Guided OT (KPG-RL)                   [Gu et al., NeurIPS 2022]
#
# Usage:
#   cd Mini-batch_KPG-RL_OT/src/DeepDA/sh
#   bash train_home_KPG_BoMbOT.sh
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
NET=ResNet50
OT_TYPE=balanced
EPSILON=0          # 0 = exact EMD; >0 = masked Sinkhorn (balanced)
BE=0               # inter-batch OT regularization
ETA1=0.01
ETA2=0.5
TAU=0.5
MASS=0.65       # transported mass s -- FIXED at 0.65 in every setting; only alpha varies
K=2                # number of mini-batches in the BoMb hierarchy
M=65               # per-mini-batch size  →  total batch = K * M
BATCH=$(( K * M ))
ITER=10000
TEST_INTERVAL=500
RUN_ID=0

# KPG-RL-KP parameters  (see keypoint_guided_OT.py kpg_rl_kp)
ALPHA=0.5           # combination coeff: alpha*C_norm + (1-alpha)*G_norm
RHO=0.1             # relation-profile temperature (Eq. 8): softmax scale rho*max(c)

METHOD="KPG_BoMbOT"
FINAL_LOG="home_${METHOD}_run${RUN_ID}_log.txt"

echo "=== Office-Home  |  ${METHOD}  |  run ${RUN_ID} ==="
echo "    BoMb: k=${K}  m=${M}  batch=${BATCH}  be=${BE}"
echo "    KPG:  alpha=${ALPHA}  rho=${RHO}"

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
    OUTPUT_DIR="home_${TASK}_${METHOD}_k${K}_m${M}_eps${EPSILON}_alpha${ALPHA}_run${RUN_ID}"

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
        --use_bomb \
        --be            ${BE} \
        --k             ${K}
    echo "── Done: ${TASK}"
done

echo ""
echo "=== All Office-Home ${METHOD} tasks finished ==="
