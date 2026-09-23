#!/bin/bash
# =============================================================================
#  STAGE 2 -- Office-Home closed-set main table (replaces the legacy Table 3)
# =============================================================================
#  WHY.  Every number in the current Table 3 was produced by the retired
#  virtual-anchor formulation (completion-plan.md §1.2), and its oracle rows are
#  inflated by 21-39 points -- confirmed in §4a against the Sec.-IV re-runs.
#  This regenerates the whole table under the formulation of Section 4, on the
#  MAIN-TABLE protocol (12 pairs, ten-crop), with the matched unguided baselines
#  the legacy table never had.
#
#  PROTOCOL (differs from Stage 1, which was a 3-pair single-crop alpha sweep):
#    m = 128, ten-crop evaluation, 10 000 iterations, eta1 0.01, eta2 0.5,
#    rho 0.1, seed 12345; balanced = exact LP, partial = exact LP with s = 0.65.
#
#  PHASES (run in order; each is self-contained and resumable)
#    A  baselines   2 regimes x 12 pairs                              =  24 runs
#    B  guided@0.5  2 regimes x 3 strategies x 12 pairs               =  72 runs
#    C  guided@1.0  2 regimes x {random, centroid} x 12 pairs         =  48 runs
#
#  Phase C exists because Stage 1 found the alpha argmax at 1.0 on this
#  benchmark -- the mask-only corner, where the guiding cost is switched off.
#  Reporting only alpha=0.5 would show the method at the worst point of its own
#  sweep; reporting only alpha=1.0 would show an m-KOT that never uses Eq. (9).
#  The table should carry both, so both are measured.
#
#  USAGE
#    bash S2-OH.sh 0                         # everything on GPU 0
#    PHASES="A" PAIRS="A2C A2P A2R C2A" bash S2-OH.sh 0     # shard by pair
#    DRY_RUN=1 bash S2-OH.sh 0               # list the runs
#    SUMMARY=1 bash S2-OH.sh                 # print the table as it fills
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICE_DIR="${SCRIPT_DIR}/../office"
GPU="${1:-0}"; RUN_ID="${RUN_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"; SUMMARY="${SUMMARY:-0}"
ITER="${ITER:-10000}"; TEST_10CROP="${TEST_10CROP:-True}"
RHO="${RHO:-0.1}"; MASS="${MASS:-0.65}"; SEED="${SEED:-12345}"
export MKUOT_AMP="${AMP:-1}" MKUOT_FAST_KP="${FAST_KP:-1}"
read -ra PHASES  <<< "${PHASES:-A B C}"
read -ra REGIMES <<< "${REGIMES:-balanced partial}"
LOG="${OFFICE_DIR}/results/S2_oh_run${RUN_ID}_log.txt"

ALL_PAIRS=(A2C A2P A2R C2A C2P C2R P2A P2C P2R R2A R2C R2P)
declare -A SRC=( [A2C]=Art [A2P]=Art [A2R]=Art [C2A]=Clipart [C2P]=Clipart [C2R]=Clipart
                 [P2A]=Product [P2C]=Product [P2R]=Product [R2A]=Real_World [R2C]=Real_World [R2P]=Real_World )
declare -A TGT=( [A2C]=Clipart [A2P]=Product [A2R]=Real_World [C2A]=Art [C2P]=Product [C2R]=Real_World
                 [P2A]=Art [P2C]=Clipart [P2R]=Real_World [R2A]=Art [R2C]=Clipart [R2P]=Product )
read -ra PAIRS <<< "${PAIRS:-${ALL_PAIRS[*]}}"

if [ "$SUMMARY" = "1" ]; then
  python3 - "$LOG" <<'PYEOF'
import re,sys,os
P=["A2C","A2P","A2R","C2A","C2P","C2R","P2A","P2C","P2R","R2A","R2C","R2P"]
d={}
if os.path.exists(sys.argv[1]):
    for line in open(sys.argv[1]):
        m=re.match(r"method snapshot/(\S+), iter: \d+, precision: ([\d.]+)",line)
        if m:
            if m.group(1) in d and d[m.group(1)]!=float(m.group(2))*100:
                sys.exit(f"DUPLICATE {m.group(1)} -- two writers; quarantine and re-run")
            d[m.group(1)]=float(m.group(2))*100
rows=[("m-OT (unguided)","balanced","base",None),("m-KOT-c","balanced","centroid","0.5"),
      ("m-KOT-r","balanced","random","0.5"),("m-KOT-f","balanced","farthest","0.5"),
      ("m-KOT-c @1.0","balanced","centroid","1.0"),("m-KOT-r @1.0","balanced","random","1.0"),
      ("m-POT (unguided)","partial","base",None),("m-KPOT-c","partial","centroid","0.5"),
      ("m-KPOT-r","partial","random","0.5"),("m-KPOT-f","partial","farthest","0.5"),
      ("m-KPOT-c @1.0","partial","centroid","1.0"),("m-KPOT-r @1.0","partial","random","1.0")]
print(f"{'Method':18s}"+"".join(f"{p:>7}" for p in P)+f"{'Avg':>8}{'n':>4}")
for lab,reg,st,a in rows:
    v=[]
    for p in P:
        k=(f"S2_oh_{p}_{reg}_m128_base_run0" if st=="base"
           else f"S2_oh_{p}_{reg}_m128_a{a}_{st}_run0")
        v.append(d.get(k))
    got=[x for x in v if x is not None]
    print(f"{lab:18s}"+"".join(f"{x:7.2f}" if x is not None else f"{'--':>7}" for x in v)
          +(f"{sum(got)/12:8.2f}" if len(got)==12 else f"{'--':>8}")+f"{len(got):4d}")
PYEOF
  exit 0
fi

LOCK_ROOT="${SCRIPT_DIR}/.s2_locks"
case "$LOCK_ROOT" in /*) ;; *) echo "LOCK_ROOT must be absolute" >&2; exit 1;; esac
mkdir -p "$LOCK_ROOT" || exit 1
CLAIMED=(); trap 'for d in "${CLAIMED[@]:-}"; do [ -n "$d" ] && rm -rf "$d"; done' EXIT INT TERM
claim () { local d="$LOCK_ROOT/$1"
  if mkdir "$d" 2>/dev/null; then echo $$ > "$d/pid"; CLAIMED+=("$d"); return 0; fi
  [ -d "$d" ] || { echo "FATAL: lock $d unusable" >&2; exit 1; }
  local o; o="$(cat "$d/pid" 2>/dev/null||true)"
  if [ -n "$o" ] && kill -0 "$o" 2>/dev/null; then return 1; fi
  echo $$ > "$d/pid"; CLAIMED+=("$d"); return 0; }

cd "$OFFICE_DIR" || exit 1
DATA_ROOT="$(cd "${SCRIPT_DIR}/../../../data" && pwd)"
export OFFICE_HOME_IMAGES_ROOT="${DATA_ROOT}/office-home"
[ -f "${DATA_ROOT}/office-home/Art.txt" ] || { echo "Office-Home lists missing" >&2; exit 1; }
mkdir -p results

run () {  # run <pair> <regime> <tag> [alpha] [strategy]
  local P=$1 REG=$2 TAG=$3 A=${4:-} ST=${5:-}
  local OUT="S2_oh_${P}_${REG}_m128_${TAG}_run${RUN_ID}"
  grep -qs "snapshot/${OUT}," "$LOG" && { echo "-- ${OUT}: done"; return; }
  [ "$DRY_RUN" = "1" ] && { echo "  DRY ${OUT}"; return; }
  claim "$OUT" || { echo "-- ${OUT}: in flight"; return; }
  local EXTRA=() KPG=()
  case "$REG" in
    balanced) EXTRA=(--ot_type balanced --epsilon 0) ;;
    partial)  EXTRA=(--ot_type partial --epsilon 0 --mass "$MASS") ;;
  esac
  [ -n "$ST" ] && KPG=(--use_kpg --alpha "$A" --kp_strategy "$ST" --kp_per_class 1 --rho "$RHO")
  echo ""; echo "== ${OUT} =="
  python train.py --gpu_id "$GPU" --net ResNet50 --dset office-home \
    --s_dset_path "${DATA_ROOT}/office-home/${SRC[$P]}.txt" \
    --t_dset_path "${DATA_ROOT}/office-home/${TGT[$P]}.txt" \
    --stratify_source --batch_size 128 --test_interval 500 --stop_step "$ITER" \
    --test_10crop "$TEST_10CROP" --seed "$SEED" \
    --output_dir "$OUT" --final_log "$LOG" \
    "${EXTRA[@]}" --eta1 0.01 --eta2 0.5 --k 1 ${KPG[@]+"${KPG[@]}"} \
    || echo "!! ${OUT} failed (status $?)"
}

for PH in "${PHASES[@]}"; do
  case "$PH" in
    A) echo "########## Phase A: matched unguided baselines ##########"
       for REG in "${REGIMES[@]}"; do for P in "${PAIRS[@]}"; do run "$P" "$REG" base; done; done ;;
    B) echo "########## Phase B: guided, alpha = 0.5 ##########"
       for REG in "${REGIMES[@]}"; do for ST in centroid random farthest; do
          for P in "${PAIRS[@]}"; do run "$P" "$REG" "a0.5_${ST}" 0.5 "$ST"; done; done; done ;;
    C) echo "########## Phase C: guided, alpha = 1.0 (mask-only corner) ##########"
       for REG in "${REGIMES[@]}"; do for ST in centroid random; do
          for P in "${PAIRS[@]}"; do run "$P" "$REG" "a1.0_${ST}" 1.0 "$ST"; done; done; done ;;
    *) echo "unknown phase $PH" >&2; exit 1 ;;
  esac
done
echo ""; echo "== S2-OH pass complete =="
