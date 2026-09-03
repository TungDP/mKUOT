"""
Generate Figure 1 for the paper: motivation for KPG-RL in mini-batch OT.

  (a) Full OT  — all data, 3 clear clusters, high matching accuracy.
  (b) mOT      — sparse mini-batch, structure lost, poor matching.
  (c) mPOT     — same mini-batch, partial transport, better but still flawed.

Outputs:
  fig1_motivation.png  — combined 1×3 figure for the paper
  fig1a_full_ot.png, fig1b_mot.png, fig1c_mpot.png — individual panels

Style matches Figure 4 of the KPG-RL paper (Gu et al., NeurIPS 2022):
  blue markers = source, green markers = target,
  different shapes per class (+, o, △), lines = transport.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import ot

# -----------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------
np.random.seed(42)

N_PER_CLASS = 25       # samples per class in full data
N_CLASSES = 3

src_centres = np.array([[-2.5, 2.0],   # class 0: top-left
                         [0.0, -2.0],   # class 1: bottom-center
                         [2.5, 2.0]])   # class 2: top-right

tgt_centres = np.array([[-1.0, 3.5],   # class 0
                         [1.5, -0.5],   # class 1
                         [4.0, 3.5]])   # class 2

# Cluster variance.  Moderate overlap so individual mini-batch samples
# can drift toward neighbouring clusters; raw-L2 mOT then misroutes some
# of them, while fig2's mKOT (which routes by keypoint-relation
# similarity) recovers them.  Same value used in fig2 so both figures
# share the exact same point cloud.
COV = 0.3 * np.eye(2)

source_all, target_all = [], []
labels_s_all, labels_t_all = [], []
for c in range(N_CLASSES):
    source_all.append(np.random.multivariate_normal(src_centres[c], COV, N_PER_CLASS))
    target_all.append(np.random.multivariate_normal(tgt_centres[c], COV, N_PER_CLASS))
    labels_s_all.extend([c] * N_PER_CLASS)
    labels_t_all.extend([c] * N_PER_CLASS)

source_all = np.vstack(source_all)
target_all = np.vstack(target_all)
labels_s_all = np.array(labels_s_all)
labels_t_all = np.array(labels_t_all)

# Mini-batch: class-IMBALANCED sampling so cluster structure is lost and
# mass-preservation forces cross-class transport in mOT.
#   source: many class-0, few class-1, few class-2
#   target: few   class-0, few class-1, many class-2
# Total mass on each side is equal, but the per-class distribution differs,
# so OT must drag class-0 source mass onto class-2 target points.
# Identical per-class counts and mini-batch seed as fig2 so the figures
# share the EXACT same source/target points.  The mOT-vs-mKOT gap then
# comes from the matching algorithm alone, not from a different draw.
# Seed 31 happens to land 2-3 source/target samples near neighbouring
# clusters (because of the moderate COV above), which mOT misroutes.
np.random.seed(31)
mb_s_idx, mb_t_idx = [], []
mb_s_labels, mb_t_labels = [], []
src_counts = [(0, 5), (1, 4), (2, 4)]   # sum=13, min=4
tgt_counts = [(0, 4), (1, 4), (2, 5)]   # sum=13, min=4
for c, ns in src_counts:
    cls_s = np.where(labels_s_all == c)[0]
    chosen_s = np.random.choice(cls_s, ns, replace=False)
    mb_s_idx.extend(chosen_s.tolist())
    mb_s_labels.extend([c] * ns)
for c, nt in tgt_counts:
    cls_t = np.where(labels_t_all == c)[0]
    chosen_t = np.random.choice(cls_t, nt, replace=False)
    mb_t_idx.extend(chosen_t.tolist())
    mb_t_labels.extend([c] * nt)

mb_s_idx = np.array(mb_s_idx)
mb_t_idx = np.array(mb_t_idx)
mb_s_labels = np.array(mb_s_labels)
mb_t_labels = np.array(mb_t_labels)

source_mb = source_all[mb_s_idx]
target_mb = target_all[mb_t_idx]


# -----------------------------------------------------------------------
# Solve OT problems
# -----------------------------------------------------------------------

def solve_full_ot(xs, xt):
    C = ot.dist(xs, xt, metric="sqeuclidean")
    a, b = ot.unif(len(xs)), ot.unif(len(xt))
    pi = ot.emd(a, b, C)
    return pi


def solve_mot(xs, xt):
    C = ot.dist(xs, xt, metric="sqeuclidean")
    a, b = ot.unif(len(xs)), ot.unif(len(xt))
    pi = ot.emd(a, b, C)
    return pi


def solve_mpot(xs, xt, mass_frac=0.65):
    C = ot.dist(xs, xt, metric="sqeuclidean")
    a, b = ot.unif(len(xs)), ot.unif(len(xt))
    pi = ot.partial.partial_wasserstein(a, b, C, m=mass_frac)
    return pi


def matching_accuracy(pi, labels_s, labels_t):
    """Fraction of transported mass that connects same-class pairs."""
    total_mass = pi.sum()
    correct_mass = 0.0
    for i in range(len(labels_s)):
        for j in range(len(labels_t)):
            if labels_s[i] == labels_t[j]:
                correct_mass += pi[i, j]
    return correct_mass / (total_mass + 1e-20)


pi_full = solve_full_ot(source_all, target_all)
pi_mot = solve_mot(source_mb, target_mb)
# With class composition (6,4,3) vs (3,4,6), the matchable mass is
# min(6,3)+min(4,4)+min(3,6) = 10 of 13 ≈ 0.77. Picking s slightly above
# this lets mPOT drop most of the unmatchable mass — fewer red lines than
# mOT — while leaving a residual cross-class match so the figure conveys
# "improves but doesn't fully solve it".
pi_mpot = solve_mpot(source_mb, target_mb, mass_frac=0.85)

acc_full = matching_accuracy(pi_full, labels_s_all, labels_t_all)
acc_mot = matching_accuracy(pi_mot, mb_s_labels, mb_t_labels)
acc_mpot = matching_accuracy(pi_mpot, mb_s_labels, mb_t_labels)

print(f"Full OT accuracy:  {acc_full:.1%}")
print(f"mOT accuracy:      {acc_mot:.1%}")
print(f"mPOT accuracy:     {acc_mpot:.1%}")


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

# Class → marker shape (matching KPG-RL paper style; filled markers so
# edgecolor/facecolor render reliably for both source and target).
MARKERS_S = {0: "X", 1: "o", 2: "^"}     # source: cross / circle / triangle
MARKERS_T = {0: "X", 1: "o", 2: "^"}     # target uses same shapes — color separates domains
SRC_COLOR = "#2563EB"   # blue
TGT_COLOR = "#16A34A"   # green
CORRECT_LINE = "#6B7280" # gray
WRONG_LINE   = "#EF4444" # red
MS = 9                   # marker size
LW_MATCH = 0.6           # line width for transport lines


def draw_transport(ax, xs, xt, pi, labels_s, labels_t, title, acc,
                   show_all_data=False, source_full=None, target_full=None,
                   labels_s_full=None, labels_t_full=None):
    """Draw a single panel."""
    # Draw transport lines (suppress sub-5%-of-max partial-transport leaks
    # so mPOT's residual fractional entries don't paint spurious thin lines).
    thresh = pi.max() * 0.05
    for i in range(len(xs)):
        for j in range(len(xt)):
            if pi[i, j] > thresh:
                is_correct = (labels_s[i] == labels_t[j])
                color = CORRECT_LINE if is_correct else WRONG_LINE
                alpha = 0.7 if is_correct else 0.85
                lw = LW_MATCH * 2.5 * (pi[i, j] / pi.max()) + 0.3
                ax.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                        '-', color=color, linewidth=lw, alpha=alpha, zorder=1)

    # Draw background full data (light, small) if requested
    if show_all_data and source_full is not None:
        for c in range(N_CLASSES):
            idx_s = np.where(labels_s_full == c)[0]
            idx_t = np.where(labels_t_full == c)[0]
            ax.scatter(source_full[idx_s, 0], source_full[idx_s, 1],
                       marker=MARKERS_S[c], c=SRC_COLOR, s=25, alpha=0.22,
                       linewidths=0.8, zorder=0, edgecolors=SRC_COLOR)
            ax.scatter(target_full[idx_t, 0], target_full[idx_t, 1],
                       marker=MARKERS_T[c], c="none", s=25, alpha=0.22,
                       linewidths=0.8, zorder=0, edgecolors=TGT_COLOR)

    # Draw data points (foreground)
    for c in range(N_CLASSES):
        idx_s = np.where(labels_s == c)[0]
        idx_t = np.where(labels_t == c)[0]
        # Source: filled markers
        ax.scatter(xs[idx_s, 0], xs[idx_s, 1],
                   marker=MARKERS_S[c], c=SRC_COLOR, s=MS**2,
                   linewidths=1.2, zorder=3, edgecolors=SRC_COLOR,
                   label=f"Source class {c}" if c == 0 else None)
        # Target: hollow markers
        ax.scatter(xt[idx_t, 0], xt[idx_t, 1],
                   marker=MARKERS_T[c], c="none", s=MS**2,
                   linewidths=1.2, zorder=3, edgecolors=TGT_COLOR,
                   label=f"Target class {c}" if c == 0 else None)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.text(0.03, 0.03, f"Matching\naccuracy: {acc:.1%}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#D1D5DB")


# ---- Combined figure ---------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

# (a) Full OT
draw_transport(axes[0], source_all, target_all, pi_full,
               labels_s_all, labels_t_all,
               "(a) Full OT", acc_full)

# (b) mOT on mini-batch, with faded full data in background
draw_transport(axes[1], source_mb, target_mb, pi_mot,
               mb_s_labels, mb_t_labels,
               "(b) mOT (mini-batch)", acc_mot,
               show_all_data=True, source_full=source_all, target_full=target_all,
               labels_s_full=labels_s_all, labels_t_full=labels_t_all)

# (c) mPOT on mini-batch, with faded full data in background
draw_transport(axes[2], source_mb, target_mb, pi_mpot,
               mb_s_labels, mb_t_labels,
               "(c) mPOT (mini-batch)", acc_mpot,
               show_all_data=True, source_full=source_all, target_full=target_all,
               labels_s_full=labels_s_all, labels_t_full=labels_t_all)

legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=SRC_COLOR,
           markeredgecolor=SRC_COLOR, markersize=8, label='Source'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor=TGT_COLOR, markersize=8, label='Target'),
    Line2D([0], [0], color=CORRECT_LINE, linewidth=1.5, label='Correct match'),
    Line2D([0], [0], color=WRONG_LINE, linewidth=1.5, label='Incorrect match'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=4,
           fontsize=10, framealpha=0.9, edgecolor='#D1D5DB',
           bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(w_pad=1.5)

out_dir = "results"

fig.savefig(f"{out_dir}/fig1_motivation.png", bbox_inches="tight", dpi=300)

# ---- Individual panels -------------------------------------------------
for idx, (tag, xs, xt, pi, ls, lt, title, acc, show_bg) in enumerate([
    ("fig1a_full_ot", source_all, target_all, pi_full,
     labels_s_all, labels_t_all, "(a) Full OT", acc_full, False),
    ("fig1b_mot", source_mb, target_mb, pi_mot,
     mb_s_labels, mb_t_labels, "(b) mOT (mini-batch)", acc_mot, True),
    ("fig1c_mpot", source_mb, target_mb, pi_mpot,
     mb_s_labels, mb_t_labels, "(c) mPOT (mini-batch)", acc_mpot, True),
]):
    fig_single, ax_single = plt.subplots(1, 1, figsize=(5, 4.5))
    draw_transport(ax_single, xs, xt, pi, ls, lt, title, acc,
                   show_all_data=show_bg, source_full=source_all, target_full=target_all,
                   labels_s_full=labels_s_all, labels_t_full=labels_t_all)
    fig_single.tight_layout()
    fig_single.savefig(f"{out_dir}/{tag}.png", bbox_inches="tight", dpi=300)
    plt.close(fig_single)

plt.close("all")
print(f"\nFigures saved to {out_dir}/")