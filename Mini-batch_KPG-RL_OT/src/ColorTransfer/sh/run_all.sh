#!/bin/bash
# Launch every colour-transfer run in parallel (one process per run spec; results
# are cached per spec under npzfiles/, so processes never write the same file).
# Usage:  bash sh/run_all.sh
set -u
source /home/doanpt/miniconda3/etc/profile.d/conda.sh
conda activate mkpg-ot
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
mkdir -p logs

PIDS=()

launch () {  # launch <src> <tgt> <spec>
  local src=$1 tgt=$2 spec=$3
  local log="logs/$(basename "$src" .bmp)_$(basename "$tgt" .bmp)_${spec//[:\/]/_}.log"
  python main.py --source "$src" --target "$tgt" --run "$spec" > "$log" 2>&1 &
  PIDS+=($!)
}

# --- pair 1: full paper-style grid ---
for spec in mOT mUOT mPOT09; do launch images/s1.bmp images/t1.bmp "$spec"; done
for s in c r f; do
  for a in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    launch images/s1.bmp images/t1.bmp "mKUOT-${s}:${a}"
  done
done

# --- pair 2: family subset ---
for spec in mOT mUOT mPOT09 mKUOT-c:0.1 mKUOT-c:0.2 mKUOT-c:0.3 mKUOT-c:0.4 mKUOT-c:0.5 mKUOT-c:0.6 mKUOT-c:0.7 mKUOT-c:0.8 mKUOT-c:0.9; do
  launch images/s2.bmp images/t2.bmp "$spec"
done

echo "launched ${#PIDS[@]} runs"
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=$((FAIL+1)); done
echo "all runs finished, failures: $FAIL"
