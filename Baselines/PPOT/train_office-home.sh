# !/bin/bash

export CUDA_VISIBLE_DEVICES=${1:-0}

DOMAINS=(Art Clipart Product Real_World)

# Resume from a specific transfer (format: "Source2Target", e.g. "Clipart2Art").
# Set to empty string to run all transfers from the beginning.
# RESUME_FROM="${RESUME_FROM:-Clipart2Art}"
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
        echo "===== Office-Home (10/5/50): $s -> $t ====="
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
            --root /home/doanpt/locnd/Mini-batch_Keypoint-Guided-Relative_OT/Baselines/PPOT/data/office-home/ 
    done
done
