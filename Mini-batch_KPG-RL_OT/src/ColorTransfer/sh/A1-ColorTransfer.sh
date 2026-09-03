#!/bin/bash
# =============================================================================
# A1 -- Guidance weight alpha, full curve (supervisor feedback #1, 2026-08-04)
# Color transfer: cheap CPU runs, so the FULL 0.1-step grid on both pairs.
# =============================================================================
# m-KUOT SETTING ONLY: mKUOT-c specs (unbalanced + plan-mined keypoints).
# alpha=1.0 is the mask-only endpoint; the m-UOT baseline is cached separately.
#
# Usage:  bash A1-ColorTransfer.sh          (CPU; runs sequentially)
# Cost:   2 pairs x 11 alpha = 22 runs (~13 min each; cached if present).
# =============================================================================
set -uo pipefail
source /home/doanpt/miniconda3/etc/profile.d/conda.sh
conda activate mkpg-ot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
export OMP_NUM_THREADS=4
mkdir -p logs

for PAIR in "images/s1.bmp images/t1.bmp" "images/s2.bmp images/t2.bmp"; do
  set -- ${PAIR}
  SRC="$1"; TGT="$2"
  # palette prep is cached; harmless to re-issue
  python main.py --source "${SRC}" --target "${TGT}" --cluster --run prep
  for ALPHA in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo "== A1 ColorTransfer $(basename ${SRC}) -> $(basename ${TGT})  alpha=${ALPHA} =="
    python main.py --source "${SRC}" --target "${TGT}" --run "mKUOT-c:${ALPHA}" \
        > "logs/A1_$(basename ${SRC} .bmp)_$(basename ${TGT} .bmp)_a${ALPHA}.log" 2>&1
  done
done
echo "A1-ColorTransfer done."
