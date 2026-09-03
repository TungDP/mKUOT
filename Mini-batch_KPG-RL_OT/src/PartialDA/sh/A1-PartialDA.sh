#!/bin/bash
# =============================================================================
# A1 -- Guidance weight alpha, full curve (supervisor feedback #1, 2026-08-04)
# Partial DA on Office-Home (65 -> 25 shared classes).
# =============================================================================
# m-KUOT SETTING ONLY: --ot_type uot + --use_kpg, practical strategy (random),
# keypoints restricted to the 25 shared classes (--n_shared_classes 25).
# 3-pair subset of the shared protocol (A2C / C2A / P2R), coarse 0.2 steps.
#
# BATCH SIZE: the plan's adopted m=64 is INFEASIBLE -- the class-balanced
# SOURCE sampler (utils.py BalancedBatchSampler) requires m >= the number of
# SOURCE classes, which is 65 even in the partial setting (only the target is
# restricted to 25).  The default here is the LITERATURE point m=65: for the
# alpha sweep the batch size only needs to be fixed, m=65 fits a single 16 GB
# GPU, and the deviation from the power-of-two rule is footnoted in the paper
# (the smallest feasible power of two, m=128, needs ~14 GB and therefore
# multi-GPU DataParallel: M=128 bash A1-PartialDA.sh 0,1,2).
#
# Usage:  bash A1-PartialDA.sh [GPU]                    (single GPU, m=65)
#         M=128 bash A1-PartialDA.sh 0,1,2              (power-of-two variant)
#         PAIRS="0 1 A2C" bash A1-PartialDA.sh 1        (single-pair quick pass)
# Cost:   3 pairs x 6 alpha = 18 runs (~11 h/pair-set on one GPU).
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
export CUDA_VISIBLE_DEVICES="${1:-0}"
KP_STRATEGY="${KP_STRATEGY:-random}"
RHO="${RHO:-0.1}"
M="${M:-65}"        # literature batch size; >= 65 source classes (see header)
N_SHARED=25

ALPHAS=(0.0 0.2 0.4 0.6 0.8 1.0)
# domain indices: 0=Art 1=Clipart 2=Product 3=RealWorld.  Override with e.g.
# PAIRS="0 1 A2C" (one pair) or PAIRS="0 1 A2C;2 3 P2R" (semicolon-separated).
IFS=';' read -ra PAIRS <<< "${PAIRS:-0 1 A2C;1 0 C2A;2 3 P2R}"

# RESUME: run_mKPOT.py appends one line per completed run to
# results/result_mkpg_uot_alpha<A>_<strategy>.txt; the key includes m, so pass
# the SAME M as the completed portion (the crashed sweep ran at M=128:
# M=128 bash A1-PartialDA.sh <gpus>).
for ALPHA in "${ALPHAS[@]}"; do
  for ENTRY in "${PAIRS[@]}"; do
    read -r SIDX TIDX TAG <<< "${ENTRY}"
    RESFILE="results/result_mkpg_uot_alpha${ALPHA}_${KP_STRATEGY}.txt"
    if [ -f "${RESFILE}" ] && grep -q "A1_mKUOT_${TAG}_m${M}_a${ALPHA}," "${RESFILE}"; then
        echo "-- A1 PartialDA ${TAG} alpha=${ALPHA}: already complete, skipping"; continue
    fi
    echo "== A1 PartialDA ${TAG}  alpha=${ALPHA} =="
    python run_mKPOT.py \
        --s "${SIDX}" --t "${TIDX}" --dset office_home --net ResNet50 \
        --batch_size "${M}" --max_iterations 5000 --test_interval 500 \
        --output "A1_mKUOT_${TAG}_m${M}_a${ALPHA}" \
        --gpu_id "${CUDA_VISIBLE_DEVICES}" \
        --ot_type uot --eta1 0.003 --eta2 0.75 --eta3 10 \
        --epsilon 0.01 --tau 0.06 --k 1 \
        --use_kpg --alpha "${ALPHA}" --kp_strategy "${KP_STRATEGY}" \
        --n_shared_classes "${N_SHARED}" --rho "${RHO}"
  done
done
echo "A1-PartialDA done."
