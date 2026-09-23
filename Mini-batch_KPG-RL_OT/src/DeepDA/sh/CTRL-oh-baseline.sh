#!/bin/bash
# Unguided Office-Home closed-set baselines at the Stage-1 operating point (m=128,
# single-crop, 10k iters).  Stage 1 runs only GUIDED arms, so without these the whole
# B1 alpha curve is uninterpretable: there is nothing to compare "guidance at its best
# alpha" against.  Stage 2 requires them anyway (completion-plan.md §3, S2-OH-base).
# Scheduling is ownership-aware: a card counts as free only when no worker of ours is
# assigned to it -- free memory alone is fooled by the keypoint-probe phase, which nearly
# collided two runs on 2026-09-16.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICE_DIR="${SCRIPT_DIR}/../office"
DATA_ROOT="$(cd "${SCRIPT_DIR}/../../../data" && pwd)"
export OFFICE_HOME_IMAGES_ROOT="${DATA_ROOT}/office-home"
export MKUOT_AMP="${AMP:-1}" MKUOT_FAST_KP="${FAST_KP:-1}"
NEED_MIB="${NEED_MIB:-13000}"
read -ra REGIMES <<< "${REGIMES:-balanced partial}"
read -ra PAIRS   <<< "${PAIRS:-C2A P2R}"
declare -A SRC=( [A2C]=Art.txt [C2A]=Clipart.txt [P2R]=Product.txt )
declare -A TGT=( [A2C]=Clipart.txt [C2A]=Art.txt [P2R]=Real_World.txt )

LOCK_ROOT="${SCRIPT_DIR}/.ctrloh_locks"; mkdir -p "$LOCK_ROOT"
CLAIMED=(); trap 'for d in "${CLAIMED[@]:-}"; do [ -n "$d" ] && rm -rf "$d"; done' EXIT INT TERM
claim () { local d="$LOCK_ROOT/$1"
  if mkdir "$d" 2>/dev/null; then echo $$ > "$d/pid"; CLAIMED+=("$d"); return 0; fi
  [ -d "$d" ] || { echo "FATAL: lock $d unusable" >&2; exit 1; }
  local o; o="$(cat "$d/pid" 2>/dev/null || true)"
  if [ -n "$o" ] && kill -0 "$o" 2>/dev/null; then return 1; fi
  echo $$ > "$d/pid"; CLAIMED+=("$d"); return 0; }

free_gpu () { local g total used
  for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    ps -eo args | grep -E 'train\.py|run_mKPOT\.py' | grep -q -- "--gpu_id ${g}\b" && continue
    read -r total used <<< "$(nvidia-smi -i "$g" --query-gpu=memory.total,memory.used \
                              --format=csv,noheader,nounits | tr -d ',')"
    [ $(( total - used )) -ge "$NEED_MIB" ] && { echo "$g"; return 0; }
  done; }

cd "$OFFICE_DIR" || exit 1
for REG in "${REGIMES[@]}"; do
  case $REG in
    balanced) EXTRA=(--ot_type balanced --epsilon 0) ;;
    partial)  EXTRA=(--ot_type partial --epsilon 0 --mass 0.65) ;;
    *) echo "unknown regime $REG" >&2; exit 1 ;;
  esac
  for P in "${PAIRS[@]}"; do
    OUT="CTRL_oh_${P}_${REG}_m128_base"
    grep -qs "snapshot/${OUT}," results/CTRL_oh_log.txt && { echo "-- $OUT: done"; continue; }
    claim "$OUT" || { echo "-- $OUT: in flight"; continue; }
    G=""; while [ -z "$G" ]; do G="$(free_gpu)"; [ -z "$G" ] && sleep 120; done
    echo ""; echo "== $OUT on GPU $G =="
    python train.py --gpu_id "$G" --net ResNet50 --dset office-home \
      --s_dset_path "${DATA_ROOT}/office-home/${SRC[$P]}" \
      --t_dset_path "${DATA_ROOT}/office-home/${TGT[$P]}" \
      --stratify_source --batch_size 128 --test_interval 500 --stop_step 10000 \
      --test_10crop False --seed 12345 \
      --output_dir "$OUT" --final_log results/CTRL_oh_log.txt \
      "${EXTRA[@]}" --eta1 0.01 --eta2 0.5 --k 1 || echo "!! $OUT failed"
  done
done
echo ""; echo "== OH baselines done =="; cat results/CTRL_oh_log.txt
