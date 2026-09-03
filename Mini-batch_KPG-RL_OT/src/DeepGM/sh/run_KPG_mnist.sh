#!/bin/bash
# KPG-RL + mini-batch OT for MNIST generative model
# Baseline: k=2, m=100, method=OT, reg=0 (Table 12 of m-POT paper)
set -e
GPU=${1:-0}
cd "$(dirname "$0")/.."

# MNIST does not support keypoint guidance (no discriminator; OT runs in flattened pixel
# space) and is not one of the paper's reported generative experiments.  This runs the
# UNGUIDED mini-batch OT baseline.
python main_mnist.py \
    --gpu-id ${GPU} \
    --method OT --reg 0 \
    --k 2 --m 100 --epochs 100 \
    --lr 0.0005 --seed 16 --latent-size 128 \
    --fid-each 5 --L 1000
