#!/bin/bash
# =============================================================================
# A1 -- Guidance weight alpha, full curve (supervisor feedback #1, 2026-08-04)
# Deep generative modeling: y-axis is FID instead of accuracy.
# =============================================================================
# m-KUOT SETTING ONLY: --method UOT + --use-kpg (keypoints are plan-mined; the
# generative code has no kp_strategy flag -- Sec. V-B protocol).  alpha is swept
# at the plan's coarse 0.2 steps; refine to 0.1 near the optimum if the curve
# is flat or bimodal (ablation-plan.md, A1 staging).
#
# Usage:  bash A1-DeepGM.sh [GPU]     (env: RUN_CIFAR/RUN_CELEBA=0|1, EPOCHS=...)
# Cost:   6 (CIFAR-10) + 6 (CelebA) = 12 runs, GPU-days scale.
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
GPU="${1:-0}"
RHO="${RHO:-0.1}"
N_KP="${N_KP:-5}"
EPOCHS="${EPOCHS:-200}"
# CelebA is DROPPED from A1 (decision 2026-08-19): at ~0.31 h/epoch it needs
# ~100 h per alpha (400 total epochs), i.e. ~30 days for the six-point sweep --
# infeasible.  The alpha curve for the generative task is reported on CIFAR-10
# alone, which is complete (6/6).  Re-enable with RUN_CELEBA=1.
RUN_CIFAR="${RUN_CIFAR:-1}"; RUN_CELEBA="${RUN_CELEBA:-0}"
# data/ lives at the repo root: src/DeepGM/sh -> ../../../data
DATA_ROOT="${DATA_ROOT:-$(cd "${SCRIPT_DIR}/../../../data" 2>/dev/null && pwd)}"
if [ -z "${DATA_ROOT}" ] || [ ! -d "${DATA_ROOT}/cifar10" ]; then
    echo "ERROR: dataset root not found (looked for '${DATA_ROOT}/cifar10')." >&2
    echo "       Set DATA_ROOT to the directory containing cifar10/ and celeba/." >&2
    exit 1
fi

ALPHAS=(0.0 0.2 0.4 0.6 0.8 1.0)

# RESUME: a run is complete when its FID csv reaches the final epoch
# (main_*.py multiplies --epochs by --k, so the last logged epoch is 2*EPOCHS-1;
# FID is evaluated every 5 epochs AND at the final epoch).  Partial runs are
# re-run from scratch: the DCGAN trainers have no mid-training checkpointing.
gm_done () {  # gm_done <csv_file> <final_epoch>
    [ -f "$1" ] && [ "$(tail -1 "$1" | cut -d, -f1)" = "$2" ]
}
FINAL_EPOCH=$(( EPOCHS * 2 - 1 ))
# --- concurrency guard -------------------------------------------------------
# A second invocation of this script (after a crash, or started on another GPU)
# would otherwise pick the same unfinished config as a live one and interleave
# writes into the same CSV / snapshot -- this actually happened to CIFAR
# alpha=1.0 on 2026-08-17.  `mkdir` is atomic, so it serves as the lock; locks
# whose owning PID is gone are reclaimed as stale.
LOCK_ROOT="${LOCK_ROOT:-.ablation_locks}"
mkdir -p "${LOCK_ROOT}"
CLAIMED=()
claim () {   # claim <key>  -> 0 if we now own it, 1 if a live process holds it
    local d="${LOCK_ROOT}/$1"
    if mkdir "${d}" 2>/dev/null; then echo $$ > "${d}/pid"; CLAIMED+=("${d}"); return 0; fi
    local owner; owner="$(cat "${d}/pid" 2>/dev/null || true)"
    if [ -n "${owner}" ] && kill -0 "${owner}" 2>/dev/null; then return 1; fi
    echo $$ > "${d}/pid"; CLAIMED+=("${d}"); return 0
}
release_all () { local d; for d in "${CLAIMED[@]:-}"; do [ -n "${d}" ] && rm -rf "${d}"; done; }
trap release_all EXIT INT TERM


if [ "$RUN_CIFAR" = "1" ]; then
for ALPHA in "${ALPHAS[@]}"; do
    CSV="csv/cifar10/Cifar10_UOT_k2_m100_reg0.01_tau1.0_mass0.65_1000_seed16_$((EPOCHS*2))epochs_kpg_nkp5_a${ALPHA}.csv"
    if gm_done "${CSV}" "${FINAL_EPOCH}"; then
        echo "-- A1 CIFAR-10 alpha=${ALPHA}: already complete, skipping"; continue
    fi
    if ! claim "gm_cifar_a${ALPHA}"; then
        echo "-- A1 CIFAR-10 alpha=${ALPHA}: IN FLIGHT in another process, skipping"; continue
    fi
    echo "== A1 DeepGM CIFAR-10  alpha=${ALPHA} =="
    python main_cifar.py \
        --gpu-id "${GPU}" --datadir "${DATA_ROOT}/cifar10" --outdir ./results \
        --method UOT --reg 0.01 --tau 1.0 \
        --m 100 --k 2 --epochs "${EPOCHS}" --fid-each 5 --seed 16 \
        --use-kpg --n-kp "${N_KP}" --alpha "${ALPHA}" --rho "${RHO}"
done
fi

if [ "$RUN_CELEBA" = "1" ]; then
for ALPHA in "${ALPHAS[@]}"; do
    CSV="csv/celeba/CelebA_UOT_k2_m200_reg0.01_tau1.0_mass0.65_1000_seed16_$((EPOCHS*2))epochs_kpg_nkp5_a${ALPHA}.csv"
    if gm_done "${CSV}" "${FINAL_EPOCH}"; then
        echo "-- A1 CelebA alpha=${ALPHA}: already complete, skipping"; continue
    fi
    if ! claim "gm_celeba_a${ALPHA}"; then
        echo "-- A1 CelebA alpha=${ALPHA}: IN FLIGHT in another process, skipping"; continue
    fi
    echo "== A1 DeepGM CelebA  alpha=${ALPHA} =="
    python main_celeba.py \
        --gpu-id "${GPU}" --datadir "${DATA_ROOT}/celeba" --outdir ./results \
        --method UOT --reg 0.01 --tau 1.0 \
        --m 200 --k 2 --epochs "${EPOCHS}" --fid-each 5 --seed 16 \
        --use-kpg --n-kp "${N_KP}" --alpha "${ALPHA}" --rho "${RHO}"
done
fi
echo "A1-DeepGM done."
