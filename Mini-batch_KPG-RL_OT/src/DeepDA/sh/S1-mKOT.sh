#!/bin/bash
# =============================================================================
#  m-KOT / m-KPOT  --  STAGE 1: guidance-weight (alpha) selection
#  completion-plan.md §3, Stage 1.        BALANCED + PARTIAL ONLY.
# =============================================================================
#  PURPOSE.  Pick the operating point alpha per (dataset, regime) on a small
#  subset, BEFORE the full Stage-2 tables are committed.  This is the step the
#  companion unbalanced paper skipped and paid for twice: its §R10d found the
#  practical margin was exactly 0.00 at alpha=0.5 -- the value its tables
#  reported -- rising monotonically to +3.40 at alpha=0.9, with the curve still
#  climbing at the edge of the grid.
#
#  The grid therefore INCLUDES alpha = 1.0, the mask-only corner.  Note that
#  corner means different things in the two regimes (completion-plan.md §1.3a):
#    balanced: keypoint mass is pinned at exactly 1/m, so alpha=1 is a fixed
#              keypoint diagonal plus ordinary OT on the (m-k)x(m-k) free block;
#    partial : keypoint mass is free and the LP drives most of it to ZERO
#              (4.4 of 5 pairs empty at s=0.65), so alpha=1 approaches m-POT on
#              a restricted support.
#  Do not expect one argmax across regimes, and do not average the two curves.
#
#  EVALUATION.  Stage 1 uses SINGLE-CROP eval (TEST_10CROP=False).  Ten-crop is
#  ~70% of a run's wall time and Stage 1 only needs the alpha ORDERING, which is
#  internally consistent either way -- in the companion paper the single-crop
#  ordering held up under ten-crop.  Stage 2 uses ten-crop for the main tables.
#  Pass TEST_10CROP=True to override.
#
#  GRID (full Stage 1 = 288 runs)
#    B1  OH     3 pairs x 3 strategies x 6 alpha x 2 regimes = 108   (office/train.py)
#    B2  PDA    3 pairs x 3 strategies x 6 alpha x 2 regimes = 108   (PartialDA/run_mKPOT.py)
#    B3  VisDA  1 pair  x 3 strategies x 6 alpha x 2 regimes =  36   (office/train.py)
#    B4  Digits S->M    x 3 strategies x 6 alpha x 2 regimes =  36   (digits/train_digits.py)
#
#  USAGE
#    bash S1-mKOT.sh 0                      # everything on GPU 0
#    RUN_B1=1 RUN_B2=0 RUN_B3=0 RUN_B4=0 bash S1-mKOT.sh 0
#    # recommended: one block per card
#    RUN_B1=1 RUN_B2=0 RUN_B3=0 RUN_B4=0 bash S1-mKOT.sh 0
#    RUN_B1=0 RUN_B2=1 RUN_B3=0 RUN_B4=0 bash S1-mKOT.sh 1
#    RUN_B1=0 RUN_B2=0 RUN_B3=1 RUN_B4=1 bash S1-mKOT.sh 2
#    # subsets
#    PAIRS="A2C" REGIMES="partial" STRATS="random" ALPHAS="0.5 1.0" bash S1-mKOT.sh 0
#    DRY_RUN=1 bash S1-mKOT.sh 0            # list what would run
#    SUMMARY=1 bash S1-mKOT.sh              # print the alpha curves
#
#  BUDGET LEVERS (completion-plan.md §3) -- pull in this order if needed:
#    STRATS="centroid random"               drop the adversarial arm  (-96 runs)
#    ALPHAS="0.5 0.7 0.9 1.0"               coarser alpha grid        (-96 runs)
#    B2_PAIRS="A2C"                         PDA on one pair           (-72 runs)
#
#  LOOP ORDER: alpha outermost, then regime, then strategy.  This makes the full
#  regime x strategy slice at a given alpha land together, so the balanced-vs-partial
#  comparison -- the paper's central mechanism claim -- is readable after the first
#  alpha completes instead of after the whole balanced arm (completion-plan.md §4a).
#
#  RESUME: every completed run is skipped via its final-log line.  Locks are
#  absolute-path and fail LOUD (completion-plan.md: port the fixed guard before
#  re-running anything -- the inert relative-path guard cost two quarantined
#  cells in the companion paper, §R10e/§R10f).
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SCRIPT_DIR}/../.."                       # .../src
OFFICE_DIR="${SCRIPT_DIR}/../office"
DIGITS_DIR="${SCRIPT_DIR}/../digits"
PDA_DIR="${REPO}/PartialDA"
GPU="${1:-0}"
RUN_ID="${RUN_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARY="${SUMMARY:-0}"

read -ra ALPHAS  <<< "${ALPHAS:-0.5 0.6 0.7 0.8 0.9 1.0}"
read -ra STRATS  <<< "${STRATS:-centroid random farthest}"
read -ra REGIMES <<< "${REGIMES:-balanced partial}"
read -ra PAIRS   <<< "${PAIRS:-A2C C2A P2R}"
read -ra B2_PAIRS <<< "${B2_PAIRS:-${PAIRS[*]}}"

TEST_10CROP="${TEST_10CROP:-False}"
ITER="${ITER:-10000}"
PDA_ITER="${PDA_ITER:-5000}"      # run_mKPOT.py budget
DIG_EPOCHS="${DIG_EPOCHS:-100}"   # train_digits.py budget (x k)
RHO="${RHO:-0.1}"
MASS="${MASS:-0.65}"
SEED="${SEED:-12345}"
export MKUOT_AMP="${AMP:-1}"
export MKUOT_FAST_KP="${FAST_KP:-1}"
RUN_B1="${RUN_B1:-1}"; RUN_B2="${RUN_B2:-1}"; RUN_B3="${RUN_B3:-1}"; RUN_B4="${RUN_B4:-1}"

LOG_DIR="${OFFICE_DIR}/results"
OH_LOG="${LOG_DIR}/S1_oh_run${RUN_ID}_log.txt"
VD_LOG="${LOG_DIR}/S1_visda_run${RUN_ID}_log.txt"

say () { printf '%s\n' "$*"; }
fail () { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ summary ---
if [ "${SUMMARY}" = "1" ]; then
  python3 - "${LOG_DIR}" "${PDA_DIR}/results" "${DIGITS_DIR}/snapshot" <<'PYEOF'
import re, os, sys, csv, glob
logdir, pdadir, digdir = sys.argv[1:4]
LINE = re.compile(r"method snapshot/(\S+), iter: \d+, precision: ([\d.]+)")
ALPHAS = ["0.5","0.6","0.7","0.8","0.9","1.0"]
def load(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            m = LINE.match(line)
            if m:
                if m.group(1) in d and d[m.group(1)] != float(m.group(2))*100:
                    sys.exit(f"DUPLICATE {m.group(1)} in {path} -- two writers; quarantine and re-run")
                d[m.group(1)] = float(m.group(2))*100
    return d
def curve(title, d, tmpl, pairs):
    rows = []
    for reg in ("balanced","partial"):
        for st in ("centroid","random","farthest"):
            vals = []
            for a in ALPHAS:
                v = [d.get(tmpl.format(p=p,a=a,st=st,reg=reg)) for p in pairs]
                vals.append(sum(v)/len(v) if all(x is not None for x in v) else None)
            if any(v is not None for v in vals):
                rows.append((reg, st, vals))
    if not rows: return
    print(f"\n=== {title} ===")
    print(f"{'regime':9s}{'strategy':10s}" + "".join(f"{'a='+a:>9}" for a in ALPHAS) + "   argmax")
    for reg, st, vals in rows:
        done = [(a,v) for a,v in zip(ALPHAS,vals) if v is not None]
        best = max(done, key=lambda t: t[1])[0] if done else "--"
        print(f"{reg:9s}{st:10s}" + "".join(f"{v:9.2f}" if v is not None else f"{'--':>9}" for v in vals)
              + f"   {best}")
oh = load(os.path.join(logdir, "S1_oh_run0_log.txt"))
curve("B1  Office-Home closed-set, m=128", oh,
      "S1_oh_{p}_{reg}_m128_a{a}_{st}_run0", ["A2C","C2A","P2R"])
vd = load(os.path.join(logdir, "S1_visda_run0_log.txt"))
curve("B3  VisDA-2017, m=64", vd, "S1_visda_T2V_{reg}_m64_a{a}_{st}_run0", ["T2V"])
# PDA: run_mKPOT appends to results/result_*.txt.  Parse the PAIR out of the run name --
# an earlier version matched on regime+alpha+strategy only and silently reported whichever
# pair happened to come last, which looks like a 3-pair mean and is not one.
print("\n=== B2  Office-Home PDA (65->25), m=128 ===")
PDA_PAIRS = ["A2C", "C2A", "P2R"]
pda = {}
for f in glob.glob(os.path.join(pdadir, "*.txt")):
    for line in open(f):
        mm = re.search(r"(S1_pda_([A-Z0-9]+)_(\w+?)_m128_a([\d.]+)_(\w+?)_run0), iter: (\d+), precision: ([\d.]+)", line)
        if mm:
            pda[(mm.group(2), mm.group(3), mm.group(4), mm.group(5))] = (float(mm.group(7))*100, int(mm.group(6)))
if not pda:
    print("  (no runs yet)")
else:
    print(f"{'regime':9s}{'strategy':10s}" + "".join(f"{'a='+a:>9}" for a in ALPHAS) + "   pairs (best-iter)")
    for reg in ("balanced","partial"):
        for st in ("centroid","random","farthest"):
            cells, notes = [], []
            for a in ALPHAS:
                vals = [pda.get((p, reg, a, st)) for p in PDA_PAIRS]
                got = [v for v in vals if v is not None]
                cells.append(sum(v[0] for v in got)/len(got) if len(got) == len(PDA_PAIRS) else None)
                notes.append(f"{len(got)}")
            if any(v is not None for v in cells) or any(n != "0" for n in notes):
                # flag runs whose BEST accuracy came at the very first eval: that is a
                # collapsing run, not a converged one.
                early = sum(1 for p in PDA_PAIRS for a in ALPHAS for st2 in (st,)
                            if pda.get((p, reg, a, st2)) and pda[(p, reg, a, st2)][1] <= 500)
                tot = sum(1 for p in PDA_PAIRS for a in ALPHAS if pda.get((p, reg, a, st)))
                print(f"{reg:9s}{st:10s}"
                      + "".join(f"{c:9.2f}" if c is not None else f"{'--':>9}" for c in cells)
                      + f"   {'/'.join(notes)}   peaked@500: {early}/{tot}")
# Digits: acc.csv per snapshot dir
print("\n=== B4  Digits SVHN->MNIST, m=512 ===")
found = False
for reg in ("balanced","partial"):
    for st in ("centroid","random","farthest"):
        cells = []
        for a in ALPHAS:
            meth, mass = ("jdot", "1") if reg == "balanced" else ("jpmbot", "0.85")
            pat = os.path.join(digdir,
                f"{meth}_svhn_to_mnist_k1_m512_lr0.0004_epsilon0.0_be0.0_mass{mass}_tau1_kpg_a{a}_{st}",
                "acc.csv")
            v = None
            if os.path.exists(pat):
                rs = list(csv.DictReader(open(pat)))
                if rs: v = max(float(r["acc"]) for r in rs)*100
            cells.append(v)
        if any(c is not None for c in cells):
            found = True
            print(f"{reg:9s}{st:10s}" + "".join(f"{c:9.2f}" if c is not None else f"{'--':>9}" for c in cells))
if not found: print("  (no runs yet)")
print()
PYEOF
  exit 0
fi

# ---------------------------------------------------------------- preflight ---
say "== preflight =="
[ -f "${OFFICE_DIR}/train.py" ]        || fail "office/train.py not found"
[ -f "${PDA_DIR}/run_mKPOT.py" ]       || fail "PartialDA/run_mKPOT.py not found"
[ -f "${DIGITS_DIR}/train_digits.py" ] || fail "digits/train_digits.py not found"
command -v python >/dev/null 2>&1      || fail "no python on PATH -- conda activate mkpg-ot"
python -c 'import torch,ot,numpy;assert torch.cuda.is_available()' 2>/dev/null \
  || fail "env unusable: need torch(+CUDA), POT, numpy -- conda activate mkpg-ot"
DATA_ROOT="${DATA_ROOT:-$(cd "${REPO}/../data" 2>/dev/null && pwd)}"
[ -n "${DATA_ROOT}" ] && [ -f "${DATA_ROOT}/office-home/Art.txt" ] \
  || fail "Office-Home lists not found under '${DATA_ROOT}/office-home'. Set DATA_ROOT."
LIST_DIR="${DATA_ROOT}/office-home"
export OFFICE_HOME_IMAGES_ROOT="${LIST_DIR}/images"
VISDA_ROOT="${VISDA_ROOT:-${DATA_ROOT}/visda-2017}"
if [ "$RUN_B3" = "1" ]; then
  [ -f "${VISDA_ROOT}/train_list.txt" ] || fail "VisDA lists not under ${VISDA_ROOT}; set VISDA_ROOT or RUN_B3=0"
  VISDA_ROOT="$(cd "${VISDA_ROOT}" && pwd)"; export VISDA_IMAGES_ROOT="${VISDA_ROOT}"
fi
mkdir -p "${LOG_DIR}"
say "  env OK, DATA_ROOT=${DATA_ROOT}, 10crop=${TEST_10CROP}, iters=${ITER}"

# --------------------------------------------------- concurrency guard --------
# Absolute, and a failure to take the lock ABORTS rather than silently granting
# ownership.  See completion-plan.md (companion §R10e): a relative LOCK_ROOT plus
# a later `cd` made this guard inert for an entire sweep.
LOCK_ROOT="${LOCK_ROOT:-${SCRIPT_DIR}/.s1_locks}"
case "${LOCK_ROOT}" in /*) ;; *) fail "LOCK_ROOT must be absolute";; esac
mkdir -p "${LOCK_ROOT}" || fail "cannot create ${LOCK_ROOT}"
CLAIMED=()
claim () {
  local d="${LOCK_ROOT}/$1"
  if mkdir "${d}" 2>/dev/null; then echo $$ > "${d}/pid"; CLAIMED+=("${d}"); return 0; fi
  [ -d "${d}" ] || fail "lock '${d}' neither creatable nor present -- guard broken, refusing to run"
  local owner; owner="$(cat "${d}/pid" 2>/dev/null || true)"
  if [ -n "${owner}" ] && kill -0 "${owner}" 2>/dev/null; then return 1; fi
  echo $$ > "${d}/pid"; CLAIMED+=("${d}"); return 0
}
release_all () { local d; for d in "${CLAIMED[@]:-}"; do [ -n "${d}" ] && rm -rf "${d}"; done; }
trap release_all EXIT INT TERM

# regime -> per-runner flags
oh_regime_flags () { case "$1" in
  balanced) echo "--ot_type balanced --epsilon 0" ;;
  partial)  echo "--ot_type partial --epsilon 0 --mass ${MASS}" ;;
  *) fail "unknown regime $1";; esac; }
pda_regime_flags () { case "$1" in
  balanced) echo "--ot_type ot  --epsilon 0 --eta1 0.001 --eta2 0.0001 --eta3 1" ;;
  partial)  echo "--ot_type pot --epsilon 0 --mass ${MASS} --eta1 0.003 --eta2 0.75 --eta3 10" ;;
  *) fail "unknown regime $1";; esac; }
dig_regime_flags () { case "$1" in
  balanced) echo "--method jdot   --epsilon 0" ;;
  partial)  echo "--method jpmbot --epsilon 0 --mass ${DIG_MASS}" ;;
  *) fail "unknown regime $1";; esac; }
# train_digits.py builds its own snapshot directory name (cfg defaults: tau=1,
# batch_epsilon=0.0, and mass=1 when --mass is not passed), so we reconstruct it
# here for the resume-skip instead of naming the run ourselves.
DIG_MASS="${DIG_MASS:-0.85}"          # SVHN->MNIST literature mass for m-POT
dig_dirname () {  # dig_dirname <regime> <alpha> <strategy>
  local meth mass
  case "$1" in balanced) meth=jdot; mass=1 ;; partial) meth=jpmbot; mass="${DIG_MASS}" ;; esac
  echo "${meth}_svhn_to_mnist_k1_m512_lr0.0004_epsilon0.0_be0.0_mass${mass}_tau1_kpg_a${2}_${3}"
}

declare -A OH_SRC=( [A2C]=Art.txt [C2A]=Clipart.txt [P2R]=Product.txt )
declare -A OH_TGT=( [A2C]=Clipart.txt [C2A]=Art.txt [P2R]=Real_World.txt )
declare -A PDA_S=( [A2C]=0 [C2A]=1 [P2R]=2 )   # 0=Art 1=Clipart 2=Product 3=RealWorld
declare -A PDA_T=( [A2C]=1 [C2A]=0 [P2R]=3 )

TOTAL=0; RAN=0; SKIP=0
plan_n () { TOTAL=$((TOTAL+1)); }

say ""
say "== Stage 1: regimes[${REGIMES[*]}] strategies[${STRATS[*]}] alphas[${ALPHAS[*]}] =="

# ------------------------------------------------------------------- B1: OH ---
if [ "$RUN_B1" = "1" ]; then
  cd "${OFFICE_DIR}" || fail "cd office"
  say "########## B1  Office-Home closed-set (m=128) ##########"
  for A in "${ALPHAS[@]}"; do for REG in "${REGIMES[@]}"; do for ST in "${STRATS[@]}"; do
    for P in "${PAIRS[@]}"; do
      OUT="S1_oh_${P}_${REG}_m128_a${A}_${ST}_run${RUN_ID}"; plan_n
      if [ -f "${OH_LOG}" ] && grep -q "snapshot/${OUT}," "${OH_LOG}"; then
        SKIP=$((SKIP+1)); continue; fi
      if [ "${DRY_RUN}" = "1" ]; then say "  DRY ${OUT}"; continue; fi
      claim "${OUT}" || { say "-- ${OUT}: in flight elsewhere"; continue; }
      say ""; say "== ${OUT} =="
      # shellcheck disable=SC2046
      python train.py --gpu_id "${GPU}" --net ResNet50 --dset office-home \
        --s_dset_path "${LIST_DIR}/${OH_SRC[$P]}" --t_dset_path "${LIST_DIR}/${OH_TGT[$P]}" \
        --stratify_source --batch_size 128 --test_interval 500 --stop_step "${ITER}" \
        --test_10crop "${TEST_10CROP}" --seed "${SEED}" \
        --output_dir "${OUT}" --final_log "${OH_LOG}" \
        $(oh_regime_flags "${REG}") --eta1 0.01 --eta2 0.5 --k 1 \
        --use_kpg --alpha "${A}" --kp_strategy "${ST}" --kp_per_class 1 --rho "${RHO}" \
        && RAN=$((RAN+1)) || say "!! ${OUT} failed (status $?)"
    done
  done; done; done
fi

# ------------------------------------------------------------------ B3: VisDA -
if [ "$RUN_B3" = "1" ]; then
  cd "${OFFICE_DIR}" || fail "cd office"
  say "########## B3  VisDA-2017 (m=64) ##########"
  for A in "${ALPHAS[@]}"; do for REG in "${REGIMES[@]}"; do for ST in "${STRATS[@]}"; do
    OUT="S1_visda_T2V_${REG}_m64_a${A}_${ST}_run${RUN_ID}"; plan_n
    if [ -f "${VD_LOG}" ] && grep -q "snapshot/${OUT}," "${VD_LOG}"; then SKIP=$((SKIP+1)); continue; fi
    if [ "${DRY_RUN}" = "1" ]; then say "  DRY ${OUT}"; continue; fi
    claim "${OUT}" || { say "-- ${OUT}: in flight elsewhere"; continue; }
    say ""; say "== ${OUT} =="
    python train.py --gpu_id "${GPU}" --net ResNet50 --dset visda \
      --s_dset_path "${VISDA_ROOT}/train_list.txt" --t_dset_path "${VISDA_ROOT}/validation_list.txt" \
      --stratify_source --batch_size 64 --test_interval 500 --stop_step "${ITER}" \
      --test_10crop "${TEST_10CROP}" --seed "${SEED}" \
      --output_dir "${OUT}" --final_log "${VD_LOG}" \
      $(oh_regime_flags "${REG}") --eta1 0.005 --eta2 1 --k 1 \
      --use_kpg --alpha "${A}" --kp_strategy "${ST}" --kp_per_class 1 --rho "${RHO}" \
      && RAN=$((RAN+1)) || say "!! ${OUT} failed (status $?)"
  done; done; done
fi

# -------------------------------------------------------------------- B2: PDA -
if [ "$RUN_B2" = "1" ]; then
  cd "${PDA_DIR}" || fail "cd PartialDA"
  mkdir -p results
  say "########## B2  Office-Home PDA 65->25 (m=128) ##########"
  for A in "${ALPHAS[@]}"; do for REG in "${REGIMES[@]}"; do for ST in "${STRATS[@]}"; do
    for P in "${B2_PAIRS[@]}"; do
      OUT="S1_pda_${P}_${REG}_m128_a${A}_${ST}_run${RUN_ID}"; plan_n
      if grep -qs "${OUT}," results/*.txt 2>/dev/null; then SKIP=$((SKIP+1)); continue; fi
      if [ "${DRY_RUN}" = "1" ]; then say "  DRY ${OUT}"; continue; fi
      claim "${OUT}" || { say "-- ${OUT}: in flight elsewhere"; continue; }
      say ""; say "== ${OUT} =="
      python run_mKPOT.py --gpu_id "${GPU}" --net ResNet50 --dset office_home \
        --s "${PDA_S[$P]}" --t "${PDA_T[$P]}" \
        --batch_size 128 --max_iterations "${PDA_ITER}" --test_interval 500 \
        --output "${OUT}" $(pda_regime_flags "${REG}") --k 1 \
        --use_kpg --alpha "${A}" --kp_strategy "${ST}" \
        --n_shared_classes 25 --rho "${RHO}" \
        && RAN=$((RAN+1)) || say "!! ${OUT} failed (status $?)"
    done
  done; done; done
fi

# ----------------------------------------------------------------- B4: Digits -
if [ "$RUN_B4" = "1" ]; then
  cd "${DIGITS_DIR}" || fail "cd digits"
  say "########## B4  Digits SVHN->MNIST (m=512) ##########"
  for A in "${ALPHAS[@]}"; do for REG in "${REGIMES[@]}"; do for ST in "${STRATS[@]}"; do
    OUT="$(dig_dirname "${REG}" "${A}" "${ST}")"; plan_n
    if [ -f "snapshot/${OUT}/final_model.pth" ]; then SKIP=$((SKIP+1)); continue; fi
    if [ "${DRY_RUN}" = "1" ]; then say "  DRY ${OUT}"; continue; fi
    claim "${OUT}" || { say "-- ${OUT}: in flight elsewhere"; continue; }
    say ""; say "== ${OUT} =="
    python train_digits.py --gpu_id "${GPU}" --source_ds svhn --target_ds mnist \
      --k 1 --mbsize 512 --n_epochs "${DIG_EPOCHS}" --test_interval 1 --nclass 10 \
      --lr 4e-4 --eta1 0.1 --eta2 0.1 --num_workers 8 --seed 1980 \
      --data_dir "${DATA_ROOT}" $(dig_regime_flags "${REG}") \
      --use_kpg --alpha "${A}" --kp_strategy "${ST}" --rho "${RHO}" \
      && RAN=$((RAN+1)) || say "!! ${OUT} failed (status $?)"
  done; done; done
fi

say ""
say "== Stage 1 pass complete: planned ${TOTAL}, already done ${SKIP}, ran ${RAN} =="
say "Curves:  SUMMARY=1 bash $(basename "${BASH_SOURCE[0]}")"
