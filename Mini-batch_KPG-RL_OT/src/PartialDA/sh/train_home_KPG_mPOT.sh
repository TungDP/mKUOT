#!/bin/bash
# m-KPOT (mini-batch Keypoint-Guided Partial OT) on Office-Home
# Partial DA setting: source = 65 classes, target = first 25 classes (labels 0-24)
#
# OT type : partial (POT)
# Baseline: m-POT (Nguyen et al., ICML 2022) uses ETA1=0.003, ETA2=0.75, ETA3=10,
#           EPSILON=0 (exact), MASS=0.65
# KPG adds: --use_kpg, ALPHA=0.5, TAU_S=TAU_T=0.5
#
# Usage:
#   bash sh/train_home_KPG_mPOT.sh [GPU_ID]
#   RESUME_FROM="CP" bash sh/train_home_KPG_mPOT.sh [GPU_ID]
#
export CUDA_VISIBLE_DEVICES=${1:-0}

OT_TYPE=pot
ETA1=0.003
ETA2=0.75
ETA3=10
EPSILON=0.01    # small entropic reg for numerical stability (0 = exact EMD)
TAU=0.06
K=1
M=65            # batch size
MASS=0.65       # transported mass s -- FIXED (the former ramp from 0.01 was removed)

KP_STRATEGY="random"
ALPHAS=(0.5 0.6 0.7 0.8 0.9)
RHO=0.1             # relation-profile temperature (Eq. 8): softmax scale rho*max(c)

N_SHARED=25     # PDA: target only has the first 25 classes (0..24) — restrict
                # KPG keypoint candidates to these classes to avoid pairing
                # source-private keypoints with mis-classified targets.

DOMAINS=(Art Clipart Product RealWorld)
DOMAIN_ABBR=(A C P R)

RESUME_FROM="${RESUME_FROM:-}"
started=0
if [ -z "$RESUME_FROM" ]; then
    started=1
fi

for ALPHA in "${ALPHAS[@]}"; do
    for s in 0 1 2 3; do
        for t in 0 1 2 3; do
            if [ "$s" -eq "$t" ]; then
                continue
            fi
            TAG="${DOMAIN_ABBR[$s]}${DOMAIN_ABBR[$t]}"
            if [ "$started" -eq 0 ]; then
                if [ "$TAG" = "$RESUME_FROM" ]; then
                    started=1
                else
                    echo "----- Skipping: ${DOMAINS[$s]} -> ${DOMAINS[$t]} (alpha=$ALPHA) -----"
                    continue
                fi
            fi
            OUTPUT="mkot_k${K}_m${M}_a${ALPHA}"
            echo "===== Office-Home PDA (65→25): ${DOMAINS[$s]} -> ${DOMAINS[$t]} (alpha=$ALPHA) ====="
            python run_mKPOT.py \
                --s "$s" \
                --t "$t" \
                --batch_size "$M" \
                --dset office_home \
                --net ResNet50 \
                --output "$OUTPUT" \
                --gpu_id "$CUDA_VISIBLE_DEVICES" \
                --ot_type "$OT_TYPE" \
                --eta1 "$ETA1" \
                --eta2 "$ETA2" \
                --eta3 "$ETA3" \
                --epsilon "$EPSILON" \
                --tau "$TAU" \
                --mass "$MASS" \
                --k "$K" \
                --use_kpg \
                --kp_strategy "$KP_STRATEGY" \
                --alpha "$ALPHA" \
                --n_shared_classes "$N_SHARED" \
                --rho ${RHO}
        done
    done
done
