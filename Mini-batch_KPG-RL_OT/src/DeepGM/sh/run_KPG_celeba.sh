#!/bin/bash
# KPG-RL + mini-batch OT for CelebA generative model
# Baseline: k=2, m=200, method=OT, reg=0 (Table 12 of m-POT paper)
set -e
GPU=${1:-1}
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${GPU}

for ALPHA in 0.5 0.6 0.7 0.8 0.9; do
    python main_celeba.py \
        --gpu-id ${GPU} \
        --method OT --reg 0 \
        --k 2 --m 200 --epochs 100 \
        --lr 0.0005 --seed 16 --latent-size 128 \
        --fid-each 5 --L 1000 \
        --use-kpg --n-kp 5 --alpha ${ALPHA} --rho 0.1
done
# main_celeba.py prints "BEST FID: <fid> @ epoch <ep>" for this run and appends
# it to csv/celeba/best_fid_summary.csv.

# ---- best FID per setting ----
echo
echo "=== CelebA: best FID per setting (sorted, lower is better) ==="
for f in csv/celeba/*.csv; do
    [ "$(basename "$f")" = "best_fid_summary.csv" ] && continue
    LC_ALL=C awk -F, -v name="$(basename "$f" .csv)" '
        NR > 1 && $2 != "" { if (min == "" || $2+0 < min+0) { min = $2; ep = $1 } }
        END { if (min != "") printf "%10.3f  @ep %-5s  %s\n", min, ep, name }
    ' "$f"
done | LC_ALL=C sort -g