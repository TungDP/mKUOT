#!/bin/bash
# KPG-RL + mini-batch OT for CIFAR-10 generative model
# Baseline: k=2, m=100, method=OT, reg=0 (Table 12 of m-POT paper)
set -e
GPU=${1:-2}
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${GPU}

for ALPHA in 0.5 0.6 0.7 0.8 0.9; do
    python main_cifar.py \
        --gpu-id ${GPU} \
        --method OT --reg 0 \
        --k 2 --m 100 --epochs 100 \
        --lr 0.0005 --seed 16 --latent-size 128 \
        --fid-each 5 --L 1000 \
        --use-kpg --n-kp 5 --alpha ${ALPHA} --rho 0.1
done
# main_cifar.py prints "BEST FID: <fid> @ epoch <ep>" for this run and appends
# it to csv/cifar10/best_fid_summary.csv.

# ---- best FID per setting ----
echo
echo "=== CIFAR-10: best FID per setting (sorted, lower is better) ==="
for f in csv/cifar10/*.csv; do
    [ "$(basename "$f")" = "best_fid_summary.csv" ] && continue
    LC_ALL=C awk -F, -v name="$(basename "$f" .csv)" '
        NR > 1 && $2 != "" { if (min == "" || $2+0 < min+0) { min = $2; ep = $1 } }
        END { if (min != "") printf "%10.3f  @ep %-5s  %s\n", min, ep, name }
    ' "$f"
done | LC_ALL=C sort -g