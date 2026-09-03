#!/bin/bash
# m-PPOT on Office-Home in CLOSED-SET DA mode (65 common / 0 source-private / 0 target-private)
#
# This script is for an apples-to-apples comparison with m-KPOT, which runs
# closed-set DA on all 12 Office-Home transfers.
#
# In closed-set mode:
#   - source and target share all 65 classes
#   - α ≈ β ≈ 1 → partial-OT degenerates to full OT
#   - L_ne (negative entropy on "unknown" samples) becomes inactive
#   - Comparable metric: per-class accuracy (NOT H-score, since there are no
#     unknown samples to classify).
#
# Important: this requires patching train.py to accept --closed-set OR setting
# `common_class=65, source_private_class=0, target_private_class=0` directly.
# The simplest patch is two lines in train.py.  See the README block below.
#
# Usage:
#   bash train_office-home_closedset.sh
#
export CUDA_VISIBLE_DEVICES=${1:-0}

DOMAINS=(Art Clipart Product Real_World)

# Resume from a specific transfer (format: "Source2Target", e.g. "Clipart2Art").
# Set to empty string to run all transfers from the beginning.
RESUME_FROM="${RESUME_FROM:-Clipart2Art}"
started=0
if [ -z "$RESUME_FROM" ]; then
    started=1
fi

for s in "${DOMAINS[@]}"; do
    for t in "${DOMAINS[@]}"; do
        if [ "$s" = "$t" ]; then
            continue
        fi
        if [ "$started" -eq 0 ]; then
            if [ "${s}2${t}" = "$RESUME_FROM" ]; then
                started=1
            else
                echo "----- Skipping already-completed transfer: $s -> $t -----"
                continue
            fi
        fi
        echo "===== Office-Home (closed-set 65/0/0): $s -> $t ====="
        python train.py \
            --task officehome \
            -s "$s" \
            -t "$t" \
            --lr 0.001 \
            --balanced \
            --mlp \
            --aug-plus \
            --cos \
            --multiprocessing-distributed \
            --root /home/doanpt/locnd/Mini-batch_Keypoint-Guided-Relative_OT/Baselines/PPOT/data/office-home/ \
            --closed-set 
    done
done
