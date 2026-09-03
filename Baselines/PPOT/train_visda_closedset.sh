#!/bin/bash
# m-PPOT on VisDA-2017 in CLOSED-SET DA mode (12 common / 0 source-private / 0 target-private)
#
# This script mirrors train_office-home_closedset.sh for an apples-to-apples
# comparison with m-KPOT, which runs closed-set DA.
#
# VisDA-2017 closed-set:
#   - single transfer: source = train (synthetic), target = validation (real)
#   - both domains share all 12 classes (labels 0..11); no unknown classes
#   - data/visda-2017/test_list.txt (the UniDA unknown set, all label 12) is
#     intentionally NOT used here.
#   - --closed-set forces common_class=12, source_private=0, target_private=0
#     (see train.py). Comparable metric: per-class / mean accuracy, NOT H-score.
#
# Usage:
#   bash train_visda_closedset.sh            # uses GPU 0
#   bash train_visda_closedset.sh 1          # uses GPU 1
#
export CUDA_VISIBLE_DEVICES=${1:-0}

echo "===== VisDA-2017 (closed-set 12/0/0): train -> validation ====="
python train.py \
    --task VisDA2017 \
    -s train \
    -t validation \
    --lr 0.0005 \
    --balanced \
    --mlp \
    --aug-plus \
    --cos \
    --multiprocessing-distributed \
    --root /home/doanpt/locnd/Mini-batch_Keypoint-Guided-Relative_OT/Baselines/PPOT/data/visda-2017/ \
    --moco-epochs 100 \
    --closed-set 

# Default moco-epochs=200