#!/bin/bash
# Progress of the OH130 Office-Home m=130 re-run (see OH130-DeepDA.sh).
# Usage: bash oh130-status.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="${SCRIPT_DIR}/../office/results"
PAIRS=(A2C A2P A2R C2A C2P C2R P2A P2C P2R R2A R2C R2P)
ALPHAS=(0.5 0.6 0.7 0.8 0.9)
printf '%-10s' "strategy"; for a in "${ALPHAS[@]}"; do printf '%8s' "a=$a"; done; echo
TOTAL=0
for s in centroid random farthest; do
    L="${R}/OH130_home_mKUOT_${s}_run0_log.txt"
    printf '%-10s' "$s"
    for a in "${ALPHAS[@]}"; do
        n=0
        [ -f "$L" ] && n=$(grep -c "_alpha${a}_${s}_run0," "$L" 2>/dev/null || true)
        printf '%6s/12' "$n"; TOTAL=$(( TOTAL + n ))
    done
    echo
done
B="${R}/OH130_home_mUOT_run0_log.txt"
nb=0; [ -f "$B" ] && nb=$(grep -c '_mUOT_m130_run0,' "$B" 2>/dev/null || true)
printf '%-10s%6s/12   (matched-m m-UOT control, no keypoints)\n' "baseline" "$nb"
echo "----------------------------------------------------------"
echo "completed: ${TOTAL}/180 guided + ${nb}/12 baseline"
echo "order per card: phase1 (a=0.5) -> baseline -> phase2 (a=0.6..0.9)"
echo -n "training now: "; pgrep -af 'train.py' | grep -oE 'OH130_home_[A-Z0-9]+_mKUOT_m130_alpha[0-9.]+_[a-z]+' | sort -u | tr '\n' ' '; echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  gpu /'
