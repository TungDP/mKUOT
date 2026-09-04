"""
Colour transfer with mini-batch keypoint-guided unbalanced optimal transport (m-KUOT).

Follows the Mini-batch-OT template pipeline (k-means palette compression, mini-batch
barycentric mapping, npz caching) and swaps the inner problem for the Sec. IV
keypoint-guided family.  Every family member — unguided m-OT / m-UOT / m-POT and the
guided m-KUOT / m-KOT / m-KPOT — runs through the SAME `transform_mKUOT` pipeline, so
any difference between panels is attributable to the inner transport problem alone.

Typical usage (run from this directory):

  # 1. compress both images once (caches k-means palettes under npzfiles/)
  python main.py --source images/s1.bmp --target images/t1.bmp --cluster --run prep

  # 2. compute any subset of runs (cached; safe to run several in parallel)
  python main.py --source images/s1.bmp --target images/t1.bmp --run mUOT,mKUOT-c:0.5

  # 3. assemble figures + W2 metrics from every cached run
  python main.py --source images/s1.bmp --target images/t1.bmp --run all --figure --palette
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte, io
from skimage.transform import resize
from sklearn import cluster

from utils import transform_mKUOT, w2_palette_distance

np.random.seed(1)

parser = argparse.ArgumentParser(description="CT with m-KUOT")
parser.add_argument("--m", type=int, default=100, metavar="N", help="mini-batch size")
parser.add_argument("--k", type=int, default=4, metavar="N", help="number of mini-batches per side")
parser.add_argument("--T", type=int, default=1000, metavar="N", help="number of iterations")
parser.add_argument("--source", type=str, metavar="N", help="Source")
parser.add_argument("--target", type=str, metavar="N", help="Target")
parser.add_argument("--cluster", action="store_true", help="Use clustering")
parser.add_argument("--palette", action="store_true", help="Save the mined keypoint colour pairs")
parser.add_argument("--figure", action="store_true", help="Assemble figures and metrics from cached runs")
parser.add_argument("--n_clusters", type=int, default=3000)
parser.add_argument("--run", type=str, default="all",
                    help="'all', 'prep', or comma list, e.g. mOT,mUOT,mPOT09,mKUOT-c:0.5")
# --- keypoint guidance (paper Sec. IV; defaults follow the DeepGM configuration) ---
parser.add_argument("--n_kp", type=int, default=5, help="number of keypoint pairs k")
parser.add_argument("--alpha", type=float, default=0.5, help="guidance weight (used when a run omits it)")
parser.add_argument("--rho", type=float, default=0.1, help="relation temperature of Eq. (8)")
parser.add_argument("--tau", type=float, default=1.0, help="marginal parameter of the KL penalties")
parser.add_argument("--reg", type=float, default=0.01, help="entropic regularisation (unbalanced runs)")
parser.add_argument("--mass", type=float, default=0.9, help="transported mass for partial runs")
parser.add_argument("--init_m", type=int, default=500, help="preliminary batch size for keypoint mining")
parser.add_argument("--n_sink", type=int, default=500, help="scaling iterations of Eq. (15)")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()


# =======================================================================
# Palette compression (verbatim template flow: k-means once, then cached)
# =======================================================================
n_clusters = args.n_clusters
name1 = args.source
name2 = args.target
source = img_as_ubyte(io.imread(name1))
target = img_as_ubyte(io.imread(name2))
reshaped_target = img_as_ubyte(resize(target, source.shape[:2]))
name1 = name1.replace("/", "")
name2 = name2.replace("/", "")
os.makedirs("npzfiles", exist_ok=True)

if args.cluster:
    X = source.reshape((-1, 3))
    source_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init=4, batch_size=100)
    source_k_means.fit(X)
    source_values = source_k_means.cluster_centers_.squeeze()
    source_labels = source_k_means.labels_
    source_compressed = source_values[source_labels]
    source_compressed.shape = source.shape
    with open("npzfiles/" + name1 + "source_compressed.npy", "wb") as f:
        np.save(f, source_compressed)
    with open("npzfiles/" + name1 + "source_values.npy", "wb") as f:
        np.save(f, source_values)
    with open("npzfiles/" + name1 + "source_labels.npy", "wb") as f:
        np.save(f, source_labels)
    np.random.seed(0)

    X = target.reshape((-1, 3))
    target_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init=4, batch_size=100)
    target_k_means.fit(X)
    target_values = target_k_means.cluster_centers_.squeeze()
    target_labels = target_k_means.labels_
    target_compressed = target_values[target_labels]
    target_compressed.shape = target.shape
    with open("npzfiles/" + name2 + "target_compressed.npy", "wb") as f:
        np.save(f, target_compressed)
    with open("npzfiles/" + name2 + "target_values.npy", "wb") as f:
        np.save(f, target_values)
    with open("npzfiles/" + name2 + "target_labels.npy", "wb") as f:
        np.save(f, target_labels)
else:
    with open("npzfiles/" + name1 + "source_values.npy", "rb") as f:
        source_values = np.load(f)
    with open("npzfiles/" + name2 + "target_values.npy", "rb") as f:
        target_values = np.load(f)
    with open("npzfiles/" + name1 + "source_labels.npy", "rb") as f:
        source_labels = np.load(f)
    with open("npzfiles/" + name2 + "target_labels.npy", "rb") as f:
        target_labels = np.load(f)

k = args.k
m = args.m
iters = args.T


# =======================================================================
# Run configurations — one string spec per family member
# =======================================================================
ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ALL_RUNS = ["mOT", "mUOT", "mPOT09"] + [
    "mKUOT-{}:{}".format(s, a) for s in ("c", "r", "f") for a in ALPHAS
]


def parse_spec(spec):
    """Map a run spec string to (tag, kwargs for transform_mKUOT)."""
    base = dict(k=k, m=m, iter=iters, rho=args.rho, tau=args.tau, mass=args.mass,
                init_m=args.init_m, n_sink=args.n_sink, seed=args.seed)
    if spec == "mOT":
        base.update(method="OT", n_kp=0, alpha=1.0, reg=0.0)
    elif spec == "mUOT":
        base.update(method="UOT", n_kp=0, alpha=1.0, reg=args.reg)
    elif spec in ("mPOT09", "mPOT099"):
        base.update(method="POT", n_kp=0, alpha=1.0, reg=0.0,
                    mass=0.9 if spec == "mPOT09" else 0.99)
    else:
        head, _, a = spec.partition(":")
        alpha = float(a) if a else args.alpha
        fam, _, strat = head.partition("-")
        method = {"mKUOT": "UOT", "mKOT": "OT", "mKPOT": "POT"}[fam]
        reg = args.reg if method == "UOT" else 0.0
        base.update(method=method, n_kp=args.n_kp, alpha=alpha, reg=reg,
                    kp_strategy=strat or "c")
    return spec.replace(":", "_a"), base


def cache_path(tag):
    return "npzfiles/CT_{}_{}_to_{}_m{}_k{}_T{}_kp{}.npz".format(
        tag, name1, name2, m, k, iters, args.n_kp)


def run_one(spec):
    tag, kw = parse_spec(spec)
    path = cache_path(tag)
    if os.path.exists(path):
        return np.load(path)
    values, img, info = transform_mKUOT(source_values, target_values, source_labels,
                                        source, desc=spec, **kw)
    np.savez(path, values=values, img=img, batch_std=info["batch_std"],
             kp_src_colors=info["kp_src_colors"], kp_tgt_colors=info["kp_tgt_colors"])
    return np.load(path)


if args.run == "prep":
    print("palette caches written for", name1, name2)
    raise SystemExit(0)

requested = ALL_RUNS if args.run == "all" else [s for s in args.run.split(",") if s]
results = {}
for spec in requested:
    results[spec] = run_one(spec)

if not args.figure:
    print("done:", ", ".join(requested))
    raise SystemExit(0)


# =======================================================================
# Metrics
#   (a) endpoint quality: exact W2^2 between transferred and target palette
#   (b) mechanism (map fidelity): RMSE between the mini-batch map and the
#       FULL-data map of the same regime — the direct measurement of
#       mini-batch-vs-full-data plan fidelity that the paper's Discussion
#       lists as untested.  Colour transfer is small enough to compute it.
# =======================================================================
import ot as pot
from kpg_ot import masked_uot_sinkhorn

os.makedirs("images/results", exist_ok=True)
pair_tag = "{}_to_{}_m{}_k{}_T{}_kp{}".format(name1, name2, m, k, iters, args.n_kp)

w_src = np.bincount(source_labels, minlength=len(source_values)).astype(float)
w_tgt = np.bincount(target_labels, minlength=len(target_values)).astype(float)


def full_reference_map(method, mass=0.9):
    """Barycentric map of the FULL n x n problem in the given regime (cached).

    Uniform marginals over the palettes, max-normalised cost — the n -> infinity
    counterpart of the mini-batch inner problem (same solver conventions).
    """
    path = "npzfiles/FULLREF_{}_{}_to_{}.npz".format(method, name1, name2)
    if os.path.exists(path):
        return np.load(path)["values"]
    s = source_values.astype(np.float64)
    t = target_values.astype(np.float64)
    n_s, n_t = len(s), len(t)
    a = np.full(n_s, 1.0 / n_s)
    b = np.full(n_t, 1.0 / n_t)
    C = pot.dist(s, t)
    Cb = C / C.max()
    if method == "OT":
        plan = pot.emd(a, b, Cb, numItermax=2000000)
    elif method == "UOT":
        plan = masked_uot_sinkhorn(Cb, np.ones_like(Cb), tau=args.tau, eps=args.reg,
                                   n_iter=2000, a=a, b=b, tol=1e-9)
    elif method == "POT":
        plan = pot.partial.partial_wasserstein(a, b, Cb, m=mass, numItermax=2000000)
    rowmass = plan.sum(axis=1)
    values = np.where(rowmass[:, None] > 1e-12,
                      plan.dot(t) / np.maximum(rowmass[:, None], 1e-12), s)
    values = np.clip(values, 0.0, 255.0)
    np.savez(path, values=values)
    return values


def map_rmse(values_a, values_b):
    """Pixel-weighted RMSE between two palette maps, RGB scaled to [0, 1]."""
    d = ((np.asarray(values_a) - np.asarray(values_b)) / 255.0) ** 2
    return float(np.sqrt((w_src @ d.sum(axis=1)) / w_src.sum()))


def mean_dispersion(z):
    """Pixel-weighted mean of the cross-batch std, RGB scaled to [0, 1]."""
    bs = np.asarray(z["batch_std"], dtype=float)
    valid = ~np.isnan(bs)
    return float((w_src[valid] @ (bs[valid] / 255.0)) / max(w_src[valid].sum(), 1e-12))


REGIME = {"mOT": "OT", "mUOT": "UOT", "mPOT09": "POT", "mPOT099": "POT"}
fullref = {}
metric_rows = [("Source (no transfer)",
                w2_palette_distance(source_values, w_src, target_values, w_tgt),
                float("nan"), float("nan"))]
for spec in requested:
    if spec in REGIME:
        method = REGIME[spec]
    else:
        method = {"mKUOT": "UOT", "mKOT": "OT", "mKPOT": "POT"}[spec.partition("-")[0]]
    if spec == "mPOT09":
        mass = 0.9
    elif spec == "mPOT099":
        mass = 0.99
    else:
        mass = args.mass
    if method not in fullref:
        fullref[method] = full_reference_map(method, mass=mass)
    metric_rows.append((spec,
                        w2_palette_distance(results[spec]["values"], w_src,
                                            target_values, w_tgt),
                        map_rmse(results[spec]["values"], fullref[method]),
                        mean_dispersion(results[spec])))

with open("images/results/CT_metrics_{}.csv".format(pair_tag), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["run", "W2sq_to_target_palette", "map_RMSE_to_fulldata_same_regime",
                 "crossbatch_std"])
    wr.writerows(metric_rows)

print("\n{:<24s} {:>12s} {:>14s} {:>12s}".format(
    "run", "W2^2->target", "RMSE->full map", "x-batch std"))
print("  (RGB scaled to [0,1]^3; lower is better for all three)")
for name, v, r, d in metric_rows:
    print("  {:<24s} {:>10.6f} {:>14s} {:>12s}".format(
        name, v,
        "-" if np.isnan(r) else "{:.6f}".format(r),
        "-" if np.isnan(d) else "{:.6f}".format(d)))


# =======================================================================
# Figures
# =======================================================================
def _panel(axis, image, title):
    axis.imshow(image)
    axis.set_title(title, fontsize=13)
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)


# --- Fig A: the family at a glance (paper-style 1-row strip).  The full-data
#     OT map is the n -> infinity reference every mini-batch member approximates.
family = [("Source", source), ("m-OT", None), ("m-UOT", None), ("m-POT s=0.9", None),
          (r"m-KUOT-c $\alpha$=0.5", None), ("Full OT (ref.)", None),
          ("Target", reshaped_target)]
family_specs = [None, "mOT", "mUOT", "mPOT09", "mKUOT-c:0.5", "FULL", None]
if all(sp in (None, "FULL") or sp in results for sp in family_specs):
    full_img = np.clip(full_reference_map("OT"), 0, 255)[source_labels] \
        .reshape(source.shape).astype("uint8")
    fig, ax = plt.subplots(1, 7, figsize=(21, 3.4))
    for i, ((title, img), sp) in enumerate(zip(family, family_specs)):
        if sp == "FULL":
            _panel(ax[i], full_img, title)
        else:
            _panel(ax[i], img if sp is None else results[sp]["img"], title)
    plt.tight_layout()
    plt.savefig("images/results/CT_family_{}.png".format(pair_tag), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

# --- Fig B: strategy x alpha image grid (Table-5 layout, qualitative) ---
strategies = [s for s in ("c", "r", "f")
              if any("mKUOT-{}:".format(s) in sp for sp in results)]
if strategies:
    fig, ax = plt.subplots(len(strategies), len(ALPHAS),
                           figsize=(3.0 * len(ALPHAS), 2.2 * len(strategies)))
    ax = np.atleast_2d(ax)
    for r, s in enumerate(strategies):
        for c, a in enumerate(ALPHAS):
            sp = "mKUOT-{}:{}".format(s, a)
            axis = ax[r][c]
            if sp in results:
                axis.imshow(results[sp]["img"])
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
            if r == 0:
                axis.set_title(r"$\alpha$={}".format(a), fontsize=12)
            if c == 0:
                axis.get_yaxis().set_visible(True)
                axis.set_yticks([])
                axis.set_ylabel("m-KUOT-{}".format(s), fontsize=12)
    plt.tight_layout()
    plt.savefig("images/results/CT_alpha_grid_{}.png".format(pair_tag), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

# --- Fig C: alpha sensitivity (paper Fig. 1 conventions; identity is carried by
#     colour AND linestyle AND marker, so no series relies on colour alone) ---
STYLE = {"c": dict(color="#2ca02c", ls="--", marker="o", label="m-KUOT-c (transport partner)"),
         "r": dict(color="#1f77b4", ls="-", marker="s", label="m-KUOT-r (random)"),
         "f": dict(color="#d62728", ls=":", marker="^", label="m-KUOT-f (farthest)")}
metric = {row[0]: row[1] for row in metric_rows}
disper = {row[0]: row[3] for row in metric_rows}
if strategies and "mUOT" in metric:
    # A visible gap between the panels and tight side margins: the paper includes this
    # at full \textwidth, so unused margin here shows up as whitespace on the page
    # (supervisor feedback 2026-09-04).  figs/make_ct_alpha.py replots the same figure
    # from the metrics CSV and must be kept in step with these settings.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    fig.subplots_adjust(wspace=0.26, left=0.065, right=0.985, bottom=0.135, top=0.91)
    panels = [(metric, r"$W_2^2$ palette distance to target", "(a) endpoint fidelity"),
              (disper, "cross-batch std of the barycentric map",
               "(b) cross-batch consistency")]
    for axis, (table, ylab, title) in zip(axes, panels):
        for s in strategies:
            xs = [a for a in ALPHAS if "mKUOT-{}:{}".format(s, a) in table]
            ys = [table["mKUOT-{}:{}".format(s, a)] for a in xs]
            if xs:
                axis.plot(xs, ys, markersize=7, linewidth=2, **STYLE[s])
        axis.axhline(table["mUOT"], color="black", ls="-.", linewidth=1.5,
                     label="m-UOT (unguided)")
        axis.set_xlabel(r"guidance weight $\alpha$ (smaller = stronger guidance)")
        axis.set_ylabel(ylab)
        axis.set_title(title, fontsize=11)
        axis.set_xticks(list(ALPHAS))
        axis.grid(axis="y", alpha=0.3, linewidth=0.5)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].legend(fontsize=9, frameon=False)
    plt.savefig("images/results/CT_alpha_sensitivity_{}.png".format(pair_tag), dpi=150)
    plt.close(fig)

# --- Fig D (--palette): the mined keypoint colour pairs ---
if args.palette and "mKUOT-c:0.5" in results:
    kp_s = results["mKUOT-c:0.5"]["kp_src_colors"]
    kp_t = results["mKUOT-c:0.5"]["kp_tgt_colors"]
    n_kp = len(kp_s)
    if n_kp:
        strip = np.zeros((2, n_kp, 3), dtype=np.uint8)
        strip[0] = np.clip(kp_s, 0, 255).astype(np.uint8)
        strip[1] = np.clip(kp_t, 0, 255).astype(np.uint8)
        fig, axis = plt.subplots(figsize=(1.0 * n_kp, 2.4))
        axis.imshow(strip, interpolation="nearest", aspect="equal")
        axis.set_yticks([0, 1])
        axis.set_yticklabels(["source kp", "target kp"], fontsize=11)
        axis.set_xticks(range(n_kp))
        axis.set_xticklabels([str(c + 1) for c in range(n_kp)], fontsize=10)
        axis.set_title("Mined keypoint pairs (strategy c)", fontsize=12)
        for side in ("top", "right", "left", "bottom"):
            axis.spines[side].set_visible(False)
        axis.tick_params(length=0)
        plt.tight_layout()
        plt.savefig("images/results/CT_keypoints_{}.png".format(pair_tag), dpi=150)
        plt.close(fig)

# --- individual BMPs, template-style ---
for spec in requested:
    io.imsave("images/results/{}_{}.bmp".format(spec.replace(":", "_a"), pair_tag),
              results[spec]["img"])

print("\nfigures + metrics written to images/results/ (tag: {})".format(pair_tag))
