#!/bin/bash
# KPG-RL + mini-batch POT (partial OT) on digits datasets
#
# Per-task MASS values follow the m-POT paper (Nguyen et al., ICML 2022),
# Appendix D.1, "Parameter settings for Digits datasets":
#   SVHN  → MNIST  : s = 0.85
#   USPS  → MNIST  : s = 0.90
#   MNIST → USPS   : s = 0.80
# Other hyperparameters (epsilon=0.1, lr=4e-4, m=500, k=1) match the baseline.
#
# Usage:
#   cd Mini-batch_KPG-RL_OT/src/DeepDA/digits
#   bash sh/train_KPG_mPOT.sh

set -e
GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU}
echo "Using GPU ${GPU}"

METHOD=jpmbot       # partial OT
K=1
M=500
EPOCH=50
TEST_INTERVAL=1
CLASS=10
EPSILON=0.1         # entropic regularization
TAU=1.0
ETA1=0.1
ETA2=0.1
LR=4e-4
SEED=1980

# KPG-RL parameters
ALPHA=0.9
RHO=0.1             # relation-profile temperature (Eq. 8): softmax scale rho*max(c)
TARGET_KEYPOINTS=random

# Per-task transported mass s for m-KPOT on Digits (this is the ONE task where s is not
# 0.65 -- Office-Home, VisDA-2017, PartialDA and DeepGM all use 0.65).
# Per-task settings: "<source>  <target>  <mass>"
TASKS=(
    "svhn   mnist  0.85"
    "usps   mnist  0.90"
    "mnist  usps   0.80"
)

for ENTRY in "${TASKS[@]}"; do
    read -r SRC TGT MASS <<< "${ENTRY}"
    echo ""
    echo "=== ${SRC} → ${TGT}  (KPG + mPOT, mass=${MASS}) ==="
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
        --data_dir /home/doanpt/locnd/Mini-batch_Keypoint-Guided-Relative_OT/Mini-batch_KPG-RL_OT/data \
        --rho ${RHO}
    echo "=== Done: ${SRC} → ${TGT} ==="
done
