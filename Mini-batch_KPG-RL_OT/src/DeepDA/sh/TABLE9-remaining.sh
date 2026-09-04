#!/bin/bash
# =============================================================================
#  m-KUOT paper -- Table 9: the experiments that our hardware cannot run
# =============================================================================
#  SELF-CONTAINED.  This is the only file you need; it calls office/train.py
#  from the repository it lives in and writes its own result logs.
#
#  WHY THIS SCRIPT EXISTS
#  ----------------------
#  Table 9 of the manuscript sweeps the two size parameters of the formulation:
#  the mini-batch size m and the number of keypoint pairs k.  Several cells are
#  empty because a 16 GB card cannot hold them (marked "oom" in the table) or
#  because we simply did not have the machine-time (marked "n/r").  This script
#  runs exactly those cells and nothing else.  Everything already measured is
#  skipped automatically, so it is safe to re-run.
#
#  WHAT IT FILLS IN
#  ----------------
#    Block A  Office-Home, m in {256, 512}, k = 65        3 pairs x 2 =  6 runs
#    Block B  VisDA-2017,  m in {256, 512}, k = 12                   =  2 runs
#    Block C  Office-Home, k in {6, 12, 24, 36} at m = 128
#                                                          3 pairs x 4 = 12 runs
#    Block D  (OPTIONAL, off by default) Office-Home, k in {130, 195, 260}
#             at m = 512 -- i.e. k ABOVE the 65-class count, which is the one
#             thing the paper says it could not test.  3 pairs x 3 =  9 runs
#
#  Total with Block D off: 20 runs.  With Block D on: 29 runs.
#
#  NOT RUNNABLE AT ALL -- please do not add these:
#    Office-Home at m in {16, 32, 64}.  Two independent reasons: the method
#    needs m > k and Office-Home has k = 65; and the class-balanced source
#    sampler raises "batch_size should be bigger than the number of classes"
#    for m < 65 (office/data_list.py).  These cells are structurally empty,
#    not merely unmeasured, and the table marks them "n/a".
#
#  HARDWARE YOU WILL NEED
#  ----------------------
#  Measured here: Office-Home at m = 128 with mixed precision = 11.3 GB.
#  Activations grow about linearly in m, so EXTRAPOLATING (not measured):
#       m = 256  ~= 22 GB  -> needs a >= 24 GB card
#       m = 512  ~= 42 GB  -> needs a >= 48 GB card (A100 80GB / H100 ideal)
#  Blocks A, B and D therefore need one large-memory GPU.  Block C is m = 128
#  and fits any 16 GB card.  If a block still runs out of memory, see
#  "IF YOU HIT OOM" at the bottom of this header.
#
#  RUNTIME
#  -------
#  Each run is 10 000 iterations plus 20 ten-crop evaluations.  On one RTX 5080
#  at m = 128 that is ~2.4 h.  Cost grows with m, so budget roughly:
#       Block A  ~6 x 5 h  = 30 h        Block C  ~12 x 2.4 h = 29 h
#       Block B  ~2 x 6 h  = 12 h        Block D  ~9 x 8 h    = 72 h
#  All four blocks are independent -- run them on different GPUs in parallel.
#
#  PREREQUISITES
#  -------------
#    * Python env with torch (CUDA), torchvision, numpy, scipy, POT, tqdm.
#      The environment we used is conda env `mkpg-ot`; activate it first.
#    * The datasets.  Set DATA_ROOT to the directory that contains:
#         office-home/Art.txt  Clipart.txt  Product.txt  Real_World.txt
#         office-home/images/...
#      and set VISDA_ROOT to a directory containing train/ , train_list.txt ,
#      validation_list.txt.  Both are checked before anything starts.
#
#  HOW TO RUN
#  ----------
#    # everything the paper needs, on GPU 0
#    DATA_ROOT=/path/to/data VISDA_ROOT=/path/to/visda-2017 \
#        bash TABLE9-remaining.sh 0
#
#    # one block at a time, on different GPUs, in parallel
#    RUN_A=1 RUN_B=0 RUN_C=0 bash TABLE9-remaining.sh 0
#    RUN_A=0 RUN_B=1 RUN_C=0 bash TABLE9-remaining.sh 1
#    RUN_A=0 RUN_B=0 RUN_C=1 bash TABLE9-remaining.sh 2
#
#    # include the k > 65 extension
#    RUN_D=1 bash TABLE9-remaining.sh 0
#
#    # see what would run, without running it
#    DRY_RUN=1 bash TABLE9-remaining.sh 0
#
#    # print the finished numbers in Table 9 layout
#    SUMMARY=1 bash TABLE9-remaining.sh
#
#  WHAT TO SEND BACK
#  -----------------
#  Just the two log files (they are plain text, a few KB):
#        <repo>/src/DeepDA/office/results/TABLE9_home_run0_log.txt
#        <repo>/src/DeepDA/office/results/TABLE9_visda_run0_log.txt
#  Or simply the output of `SUMMARY=1 bash TABLE9-remaining.sh`.
#
#  IF YOU HIT OOM
#  --------------
#    1. AMP is already on.  AMP=0 turns it off (uses MORE memory; don't).
#    2. TEST_10CROP=False cuts evaluation memory and ~70% of the runtime, but
#       then the numbers are single-crop and NOT comparable to the rest of
#       Table 9 -- only use this to smoke-test that a config starts.
#    3. Multi-GPU: pass a comma list, e.g. `bash TABLE9-remaining.sh 0,1`.
#       train.py will DataParallel across them.  WARNING: we measured this to
#       be ~80x slower than a single card at these batch sizes, so prefer one
#       big GPU over several small ones.
#    4. Reduce --kp_probe (default 4096) if the keypoint probe itself OOMs.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICE_DIR="${SCRIPT_DIR}/../office"
GPU="${1:-0}"
RUN_ID="${RUN_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARY="${SUMMARY:-0}"

# Fixed experimental settings -- these match the rest of Table 9 and must not
# be changed, or the new cells stop being comparable with the measured ones.
ITER="${ITER:-10000}"
TEST_10CROP="${TEST_10CROP:-True}"      # ten-crop, as in the main tables
ALPHA="${ALPHA:-0.5}"                   # Table 9 is measured at alpha = 0.5
RHO="${RHO:-0.1}"
KP_STRATEGY="${KP_STRATEGY:-random}"    # the practical variant
SEED="${SEED:-12345}"
export MKUOT_AMP="${AMP:-1}"            # mixed precision (halves activations)
export MKUOT_FAST_KP="${FAST_KP:-1}"    # cached keypoint batching; big speed-up

RUN_A="${RUN_A:-1}"; RUN_B="${RUN_B:-1}"; RUN_C="${RUN_C:-1}"; RUN_D="${RUN_D:-0}"

OH_LOG="${OFFICE_DIR}/results/TABLE9_home_run${RUN_ID}_log.txt"
VD_LOG="${OFFICE_DIR}/results/TABLE9_visda_run${RUN_ID}_log.txt"

# ----------------------------------------------------------------- summary ---
if [ "${SUMMARY}" = "1" ]; then
    python3 - "$OH_LOG" "$VD_LOG" <<'PYEOF'
import re, sys, os
LINE = re.compile(r"method snapshot/(\S+), iter: \d+, precision: ([\d.]+)")
res = {}
for path in sys.argv[1:]:
    if not os.path.exists(path):
        continue
    for line in open(path):
        m = LINE.match(line)
        if m:
            res[m.group(1)] = float(m.group(2)) * 100.0
PAIRS = ["A2C", "C2A", "P2R"]
def oh(tag):
    v = [res.get(f"TABLE9_office-home_{p}_mKUOT_{tag}_run0") for p in PAIRS]
    return (sum(v)/3, v) if all(x is not None for x in v) else (None, v)
print("\n=== Table 9, panel (a): mini-batch size m ===")
print(f"{'m':>6}  {'Office-Home (k=65)':>20}  {'VisDA-2017 (k=12)':>19}")
for m in (256, 512):
    a, _ = oh(f"m{m}_pc1")
    vk = f"TABLE9_visda_T2V_mKUOT_m{m}_pc1_run0"
    b = res.get(vk)
    print(f"{m:>6}  {(f'{a:.2f}' if a else 'pending'):>20}  {(f'{b:.2f}' if b else 'pending'):>19}")
print("\n=== Table 9, panel (b): keypoint pairs k, Office-Home at m = 128 ===")
print(f"{'k':>6}  {'A2C':>7} {'C2A':>7} {'P2R':>7}  {'mean':>7}")
for k in (6, 12, 24, 36):
    a, v = oh(f"m128_nc{k}_pc1")
    cells = "  ".join(f"{x:7.2f}" if x is not None else f"{'--':>7}" for x in v)
    print(f"{k:>6}  {cells}  {(f'{a:7.2f}' if a else f'{chr(45)*2:>7}')}")
print("\n=== Optional block D: k above the 65-class count, m = 512 ===")
for nc, pc in ((65, 2), (65, 3), (65, 4)):
    a, v = oh(f"m512_nc{nc}_pc{pc}")
    print(f"  k = {nc*pc:>3} (pc={pc})  " +
          ("  ".join(f"{x:.2f}" if x is not None else "--" for x in v)) +
          f"   mean {(f'{a:.2f}' if a else 'pending')}")
print()
PYEOF
    exit 0
fi

# ---------------------------------------------------------------- preflight ---
say () { printf '%s\n' "$*"; }
fail () { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

say "== preflight =="
[ -f "${OFFICE_DIR}/train.py" ] || fail "train.py not found at ${OFFICE_DIR}. Keep this script inside the repository (src/DeepDA/sh/)."

command -v python >/dev/null 2>&1 || fail "no 'python' on PATH -- activate your environment first (e.g. conda activate mkpg-ot)."
python - <<'PYEOF' || fail "the Python environment is not usable (see message above)."
import sys
try:
    import torch, torchvision, numpy, tqdm
except Exception as e:
    sys.exit(f"missing package: {e}")
if not torch.cuda.is_available():
    sys.exit("torch reports no CUDA device")
n = torch.cuda.device_count()
free, total = torch.cuda.mem_get_info(0)
print(f"  torch {torch.__version__}, {n} CUDA device(s); "
      f"device 0 has {total/2**30:.1f} GiB total, {free/2**30:.1f} GiB free")
if total / 2**30 < 20:
    print("  NOTE: <20 GiB on device 0 -- blocks A, B and D will very likely OOM.")
    print("        Block C (m=128) is fine. See 'IF YOU HIT OOM' in the header.")
PYEOF

# Datasets.  Both roots are overridable; we check for the exact files used.
DATA_ROOT="${DATA_ROOT:-$(cd "${SCRIPT_DIR}/../../../data" 2>/dev/null && pwd)}"
[ -n "${DATA_ROOT}" ] || fail "DATA_ROOT is unset and the default location does not exist. Set DATA_ROOT=/path/to/data."
LIST_DIR="${DATA_ROOT}/office-home"
if [ "$RUN_A$RUN_C$RUN_D" != "000" ]; then
    for f in Art.txt Clipart.txt Product.txt Real_World.txt; do
        [ -f "${LIST_DIR}/${f}" ] || fail "missing ${LIST_DIR}/${f}. Set DATA_ROOT to the directory containing office-home/."
    done
    [ -d "${LIST_DIR}/images" ] || say "  WARNING: ${LIST_DIR}/images not found; the list files must then contain absolute image paths."
    say "  Office-Home OK: ${LIST_DIR}"
fi
VISDA_ROOT="${VISDA_ROOT:-${SCRIPT_DIR}/../../../../Baselines/Mini-batch-OT/DeepDA/office/data/visda-2017}"
if [ "$RUN_B" = "1" ]; then
    [ -f "${VISDA_ROOT}/train_list.txt" ] && [ -f "${VISDA_ROOT}/validation_list.txt" ] \
        || fail "VisDA lists not found under ${VISDA_ROOT}. Set VISDA_ROOT, or pass RUN_B=0 to skip VisDA."
    VISDA_ROOT="$(cd "${VISDA_ROOT}" && pwd)"
    say "  VisDA-2017 OK: ${VISDA_ROOT}"
fi

export OFFICE_HOME_IMAGES_ROOT="${LIST_DIR}/images"
export VISDA_IMAGES_ROOT="${VISDA_ROOT:-}"
mkdir -p "${OFFICE_DIR}/results" || fail "cannot write to ${OFFICE_DIR}/results"
say "  writing logs to ${OFFICE_DIR}/results/TABLE9_*_run${RUN_ID}_log.txt"

# --------------------------------------------------- concurrency guard -------
# Absolute path on purpose: we cd into office/ below, and a relative lock root
# would silently stop working there (that bug cost us a sweep on 2026-09-03).
LOCK_ROOT="${LOCK_ROOT:-${SCRIPT_DIR}/.table9_locks}"
case "${LOCK_ROOT}" in /*) ;; *) fail "LOCK_ROOT must be an absolute path";; esac
mkdir -p "${LOCK_ROOT}" || fail "cannot create ${LOCK_ROOT}"
CLAIMED=()
claim () {
    local d="${LOCK_ROOT}/$1"
    if mkdir "${d}" 2>/dev/null; then echo $$ > "${d}/pid"; CLAIMED+=("${d}"); return 0; fi
    # Only an existing directory means "held".  Anything else means the guard
    # is broken, and continuing would risk two writers on one config.
    [ -d "${d}" ] || fail "lock '${d}' can neither be created nor found; refusing to run unprotected."
    local owner; owner="$(cat "${d}/pid" 2>/dev/null || true)"
    if [ -n "${owner}" ] && kill -0 "${owner}" 2>/dev/null; then return 1; fi
    echo $$ > "${d}/pid"; CLAIMED+=("${d}"); return 0
}
release_all () { local d; for d in "${CLAIMED[@]:-}"; do [ -n "${d}" ] && rm -rf "${d}"; done; }
trap release_all EXIT INT TERM

cd "${OFFICE_DIR}" || fail "cannot enter ${OFFICE_DIR}"

# ------------------------------------------------------------------ runner ---
# run_one <dset> <s_list> <t_list> <task> <m> <eta1> <eta2> <tau> <pc> <nc|-> <log>
run_one () {
    local DSET=$1 SPATH=$2 TPATH=$3 TASK=$4 M=$5 ETA1=$6 ETA2=$7 TAU=$8 PC=$9 NC=${10} LOG=${11}
    local NC_FLAGS=() KTAG
    if [ "$NC" != "-" ]; then NC_FLAGS=(--kp_n_classes "${NC}"); KTAG="m${M}_nc${NC}_pc${PC}"
    else KTAG="m${M}_pc${PC}"; fi
    local OUT="TABLE9_${DSET}_${TASK}_mKUOT_${KTAG}_run${RUN_ID}"

    if [ -f "${LOG}" ] && grep -q "snapshot/${OUT}," "${LOG}"; then
        say "-- ${OUT}: already complete, skipping"; return 0
    fi
    if [ "${DRY_RUN}" = "1" ]; then say "DRY-RUN would run: ${OUT}"; return 0; fi
    if ! claim "${OUT}"; then say "-- ${OUT}: in flight elsewhere, skipping"; return 0; fi

    say ""
    say "===================================================================="
    say "  ${OUT}   (GPU ${GPU})"
    say "===================================================================="
    python train.py \
        --gpu_id "${GPU}" --net ResNet50 --dset "${DSET}" \
        --s_dset_path "${SPATH}" --t_dset_path "${TPATH}" \
        --stratify_source --batch_size "${M}" \
        --test_interval 500 --stop_step "${ITER}" --test_10crop "${TEST_10CROP}" \
        --output_dir "${OUT}" --final_log "${LOG}" --seed "${SEED}" \
        --ot_type unbalanced --eta1 "${ETA1}" --eta2 "${ETA2}" \
        --epsilon 0.01 --tau "${TAU}" --k 1 \
        --use_kpg --alpha "${ALPHA}" --kp_strategy "${KP_STRATEGY}" \
        --kp_per_class "${PC}" ${NC_FLAGS[@]+"${NC_FLAGS[@]}"} --rho "${RHO}"
    local rc=$?
    [ $rc -ne 0 ] && say "!! ${OUT} exited with status ${rc} (likely OOM; see header)"
    return 0        # keep going: one failed cell must not abort the rest
}

# Office-Home hyperparameters (Table 1 of the paper): eta1 0.01, eta2 0.5, tau 0.5
OH_PAIRS=( "Art.txt Clipart.txt A2C" "Clipart.txt Art.txt C2A" "Product.txt Real_World.txt P2R" )
oh_run () {   # oh_run <m> <pc> <nc|->
    for E in "${OH_PAIRS[@]}"; do
        read -r S T TAG <<< "${E}"
        run_one office-home "${LIST_DIR}/${S}" "${LIST_DIR}/${T}" "${TAG}" "$1" 0.01 0.5 0.5 "$2" "$3" "${OH_LOG}"
    done
}

say ""
say "== plan =="
[ "$RUN_A" = "1" ] && say "  Block A: Office-Home m in {256,512}, k=65      (6 runs)"
[ "$RUN_B" = "1" ] && say "  Block B: VisDA-2017  m in {256,512}, k=12      (2 runs)"
[ "$RUN_C" = "1" ] && say "  Block C: Office-Home k in {6,12,24,36}, m=128  (12 runs)"
[ "$RUN_D" = "1" ] && say "  Block D: Office-Home k in {130,195,260}, m=512 (9 runs)"
say ""

# ---- Block A: the "oom" cells of panel (a), Office-Home -------------------
if [ "$RUN_A" = "1" ]; then
    say "########## Block A -- Office-Home, larger mini-batches ##########"
    for M in 256 512; do oh_run "${M}" 1 "-"; done
fi

# ---- Block B: the "n/r" cells of panel (a), VisDA -------------------------
if [ "$RUN_B" = "1" ]; then
    say "########## Block B -- VisDA-2017, larger mini-batches ##########"
    # VisDA hyperparameters (Table 1): eta1 0.005, eta2 1, tau 0.3
    for M in 256 512; do
        run_one visda "${VISDA_ROOT}/train_list.txt" "${VISDA_ROOT}/validation_list.txt" \
                T2V "${M}" 0.005 1 0.3 1 "-" "${VD_LOG}"
    done
fi

# ---- Block C: the missing k values of panel (b), Office-Home -------------
# k = kp_n_classes x kp_per_class, so k=6 means "one pair for each of 6 classes".
if [ "$RUN_C" = "1" ]; then
    say "########## Block C -- Office-Home, fewer keypoint pairs ##########"
    for NC in 6 12 24 36; do oh_run 128 1 "${NC}"; done
fi

# ---- Block D (optional): k ABOVE the class count -------------------------
# The paper states it cannot test k > 65 on Office-Home, because k < m is
# required and m was capped at 128.  At m = 512 that limit disappears:
# kp_per_class = 2, 3, 4 over all 65 classes gives k = 130, 195, 260.
# This is the only block that answers a stated limitation rather than filling
# a blank, which is why it is opt-in.
if [ "$RUN_D" = "1" ]; then
    say "########## Block D -- Office-Home, k above the class count ##########"
    for PC in 2 3 4; do oh_run 512 "${PC}" 65; done
fi

say ""
say "== done =="
say "Print the results with:   SUMMARY=1 bash $(basename "${BASH_SOURCE[0]}")"
say "Send back:               ${OH_LOG}"
[ "$RUN_B" = "1" ] && say "                         ${VD_LOG}"
