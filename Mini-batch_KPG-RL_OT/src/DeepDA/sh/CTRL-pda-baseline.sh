#!/bin/bash
# =============================================================================
#  PDA unguided baseline control  --  is the partial-PDA pipeline usable at all?
# =============================================================================
#  WHY.  Stage 1 runs only GUIDED arms, so when m-KPOT on PDA came back at
#  26-37 % with most runs peaking at the first evaluation (completion-plan.md
#  §4b), there was no way to tell whether guidance was failing or the PDA
#  configuration is simply broken at these hyperparameters.  The two regimes use
#  wildly different weights -- eta2 0.0001 vs 0.75, eta3 1 vs 10, both inherited
#  from the retired formulation -- so this is a real possibility.  The companion
#  paper hit exactly this and had to discard its PDA runs entirely (its §R9).
#
#  WHAT.  The same pipeline with NO keypoints anywhere (no --use_kpg): no Eq.(5)
#  injection, no mask, no guiding cost.  3 pairs x 2 regimes = 6 runs.  Partial
#  is the one that was approved; balanced is added because two of its three
#  guided arms collapse too, so it needs the same reference -- and both are
#  required for Stage 2's matched baselines regardless (§3, S2-PDA-base).
#
#  SCHEDULING.  All three cards are busy with Stage 1, so this polls for a card
#  with enough free memory and takes it, rather than disturbing a running chain.
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDA_DIR="${SCRIPT_DIR}/../../PartialDA"
NEED_MIB="${NEED_MIB:-13000}"      # a PDA run was measured at 12.6 GiB with AMP
export MKUOT_AMP="${AMP:-1}"
read -ra REGIMES <<< "${REGIMES:-partial balanced}"
read -ra PAIRS   <<< "${PAIRS:-A2C C2A P2R}"
declare -A S=( [A2C]=0 [C2A]=1 [P2R]=2 )
declare -A T=( [A2C]=1 [C2A]=0 [P2R]=3 )

LOCK_ROOT="${SCRIPT_DIR}/.ctrl_locks"
mkdir -p "${LOCK_ROOT}" || { echo "cannot create ${LOCK_ROOT}" >&2; exit 1; }
CLAIMED=(); trap 'for d in "${CLAIMED[@]:-}"; do [ -n "$d" ] && rm -rf "$d"; done' EXIT INT TERM
claim () { local d="${LOCK_ROOT}/$1"
  if mkdir "$d" 2>/dev/null; then echo $$ > "$d/pid"; CLAIMED+=("$d"); return 0; fi
  [ -d "$d" ] || { echo "FATAL: lock $d unusable" >&2; exit 1; }
  local o; o="$(cat "$d/pid" 2>/dev/null || true)"
  if [ -n "$o" ] && kill -0 "$o" 2>/dev/null; then return 1; fi
  echo $$ > "$d/pid"; CLAIMED+=("$d"); return 0; }

free_gpu () {   # a card is free only if NOTHING of ours is assigned to it.
  # Free memory alone is NOT sufficient: a run in its keypoint-probe phase is
  # CPU-bound and holds ~1.3 GiB, so the card looks idle and then claims 12.6 GiB
  # minutes later.  That nearly collided a control run with a Stage-1 run on
  # 2026-09-16.  Check process ownership, which is exact, and memory as a backstop.
  local g used total
  for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    if ps -eo args | grep -E 'train\.py|run_mKPOT\.py' | grep -q -- "--gpu_id ${g}\b"; then
      continue
    fi
    read -r total used <<< "$(nvidia-smi -i "$g" --query-gpu=memory.total,memory.used \
                              --format=csv,noheader,nounits | tr -d ',')"
    if [ $(( total - used )) -ge "${NEED_MIB}" ]; then echo "$g"; return 0; fi
  done
}

cd "${PDA_DIR}" || { echo "cannot cd ${PDA_DIR}" >&2; exit 1; }
mkdir -p results
for REG in "${REGIMES[@]}"; do
  case "$REG" in
    partial)  FLAGS=(--ot_type pot --epsilon 0 --mass 0.65 --eta1 0.003 --eta2 0.75 --eta3 10) ;;
    balanced) FLAGS=(--ot_type ot  --epsilon 0 --eta1 0.001 --eta2 0.0001 --eta3 1) ;;
    *) echo "unknown regime $REG" >&2; exit 1 ;;
  esac
  for P in "${PAIRS[@]}"; do
    OUT="CTRL_pda_${P}_${REG}_m128_base"
    grep -qs "${OUT}," results/*.txt && { echo "-- ${OUT}: done, skipping"; continue; }
    claim "${OUT}" || { echo "-- ${OUT}: in flight elsewhere"; continue; }
    G=""; while [ -z "$G" ]; do G="$(free_gpu)"; [ -z "$G" ] && sleep 120; done
    echo ""; echo "== ${OUT}  on GPU ${G} =="
    python run_mKPOT.py --gpu_id "${G}" --net ResNet50 --dset office_home \
      --s "${S[$P]}" --t "${T[$P]}" \
      --batch_size 128 --max_iterations 5000 --test_interval 500 \
      --output "${OUT}" "${FLAGS[@]}" --k 1 --n_shared_classes 25 \
      || echo "!! ${OUT} failed (status $?)"
  done
done
echo ""; echo "== PDA baseline control done =="
grep -h 'CTRL_pda' results/*.txt 2>/dev/null | sed 's/method //; s/, iter: /  it=/; s/, precision: /  /' | sort
