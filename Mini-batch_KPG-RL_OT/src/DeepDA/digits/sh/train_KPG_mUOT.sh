#!/bin/bash
# KPG-RL + mini-batch UOT (unbalanced, Sinkhorn) on digits datasets
#
# Hyperparameters follow the baseline exp_mUOT.sh (epsilon=0.1, tau=1.0)
# plus KPG-RL defaults.
#
# Usage:
#   cd Mini-batch_KPG-RL_OT/src/DeepDA/digits
#   bash sh/exp_KPG_mUOT.sh

set -e
GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU}
echo "Using GPU ${GPU}"

METHOD=jumbot       # unbalanced OT
K=1
M=500
EPOCH=100
TEST_INTERVAL=1
CLASS=10
EPSILON=0.1         # Sinkhorn regularization (required >0 for unbalanced)
TAU=1.0             # marginal relaxation
ETA1=0.1
ETA2=0.1
MASS=0.85       # unused by unbalanced OT (kept for interface consistency)
LR=4e-4
SEED=1980

# KPG-RL parameters
ALPHA=0.5
RHO=0.1             # relation-profile temperature (Eq. 8): softmax scale rho*max(c)
TARGET_KEYPOINTS=centroid

TASKS=(
    "svhn   mnist"
    "usps   mnist"
    "mnist  usps"
)

for ENTRY in "${TASKS[@]}"; do
    read -r SRC TGT <<< "${ENTRY}"
    echo ""
    echo "=== ${SRC} → ${TGT}  (KPG + mUOT) ==="
    python train_digits.py \
        --gpu_id    ${GPU} \
        --method    ${METHOD} \
        --source_ds ${SRC} \
        --target_ds ${TGT} \
        --k         ${K} \
        --mbsize    ${M} \
        --n_epochs  ${EPOCH} \
        --test_interval ${TEST_INTERVAL} \
        --nclass    ${CLASS} \
        --epsilon   ${EPSILON} \
        --tau       ${TAU} \
        --mass      ${MASS} \
        --lr        ${LR} \
        --eta1      ${ETA1} \
        --eta2      ${ETA2} \
        --num_workers 8 \
        --seed      ${SEED} \
        --use_kpg \
        --kp_strategy ${TARGET_KEYPOINTS} \
        --alpha     ${ALPHA} \
        --data_dir /workspace/Mini-batch_Keypoint-Guided-Relative_OT/Mini-batch_KPG-RL_OT/data \
        --rho ${RHO}
    echo "=== Done: ${SRC} → ${TGT} ==="
done
