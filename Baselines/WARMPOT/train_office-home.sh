#!/bin/bash
# WARMPOT on Office-Home in PARTIAL DA mode (65 source / 25 target classes)
#
# PDA setting: source has all 65 classes, target has only the first 25
# alphabetically sorted classes (Alarm_Clock ... Flip_Flops).
# Metric: per-class accuracy on the 25 target classes.
#
# Hyperparameters from Table E1 (paper appendix) and run.sh:
#   batch_size=65, eta1=0.5, eta2=7.0, eta3=0.25, epsilon=7.0
#   mass=0.8, beta=0.35, max_iterations=5000, seed=2020
#
# Usage:
#   bash train_office-home.sh [GPU_ID]
#   RESUME_FROM="Cl2Pr" bash train_office-home.sh [GPU_ID]
#
export CUDA_VISIBLE_DEVICES=${1:-0}

# Domain index mapping (warmpot.py uses indices 0-3):
#   0=Art(Ar), 1=Clipart(Cl), 2=Product(Pr), 3=Real_World(Rw)
DOMAIN_NAMES=(Art Clipart Product Real_World)

# Logging: per-transfer full stdout under LOG_DIR, summary in RESULT_FILE.
LOG_DIR="logs/office-home"
RESULT_FILE="result_warmpot.txt"
mkdir -p "$LOG_DIR"

# Resume from a specific transfer (format: "Source2Target", e.g. "Cl2Pr").
# Set to empty string to run all transfers from the beginning.
RESUME_FROM="${RESUME_FROM:-}"
started=0
if [ -z "$RESUME_FROM" ]; then
    started=1
fi

for s in 0 1 2 3; do
    for t in 0 1 2 3; do
        if [ "$s" -eq "$t" ]; then
            continue
        fi
        S_NAME=${DOMAIN_NAMES[$s]}
        T_NAME=${DOMAIN_NAMES[$t]}
        TAG="${S_NAME:0:2}2${T_NAME:0:2}"
        if [ "$started" -eq 0 ]; then
            if [ "$TAG" = "$RESUME_FROM" ]; then
                started=1
            else
                echo "----- Skipping already-completed transfer: $S_NAME -> $T_NAME -----"
                continue
            fi
        fi
        echo "===== Office-Home (PDA 65/25): $S_NAME -> $T_NAME ====="
        LOG_FILE="${LOG_DIR}/${TAG}.log"
        python -u warmpot.py \
            --dset OfficeHome \
            --s "$s" \
            --t "$t" \
            --gpu_id "${CUDA_VISIBLE_DEVICES}" \
            --batch_size 65 \
            --eta1 0.5 \
            --eta2 7.0 \
            --eta3 0.25 \
            --epsilon 7.0 \
            --mass 0.65 \
            --beta 1.00 \
            --max_iterations 5000 \
            --test_interval 100 \
            --seed 2020 \
            --mass_increase_i 2500 \
            --net ResNet50 \
            --use_wandb 0 2>&1 | tee "$LOG_FILE"

        # Append final best accuracy to the summary result file.
        BEST_ACC=$(grep -oE "Acc:[[:space:]]*[0-9.]+" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
        echo "method warmpot_${TAG} (${S_NAME}->${T_NAME}), best_acc: ${BEST_ACC:-N/A}" >> "$RESULT_FILE"
    done
done
