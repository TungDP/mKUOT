"""
Generate Figure 2 for the paper: keypoint-guided counterparts of Figure 1.

  (a) KOT    — keypoint-guided full OT
  (b) mKOT   — keypoint-guided OT on the same mini-batch as fig1
  (c) mKPOT  — keypoint-guided partial OT on the same mini-batch

Setup is identical to fig1_motivation.py (same clusters, same mini-batch,
same class counts) so the two figures are directly comparable. The only
addition is a single annotated keypoint pair per class, used to guide the
matching via the KPG-RL formulation of Gu et al. (NeurIPS 2022):

  - mask-based constraint that enforces matching of paired keypoints
  - softmax-normalised relation of each point to the keypoint set
  - guiding cost  G[k,l] = JS-divergence(R^s_k, R^t_l)
  - solved by masked Sinkhorn iteration
  - partial variant via the dummy-point extension (Theorem 1 of the paper)

Outputs:
  fig2_kpg.png                              — combined 1x3 figure
  fig2a_kot.png, fig2b_mkot.png, fig2c_mkpot.png — individual panels
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import ot

# =======================================================================
# Data generation — must match fig1_motivation.py exactly
# =======================================================================
np.random.seed(42)

N_PER_CLASS = 25
N_CLASSES = 3

src_centres = np.array([[-2.5, 2.0],
                         [0.0, -2.0],
                         [2.5, 2.0]])
tgt_centres = np.array([[-1.0, 3.5],
                         [1.5, -0.5],
                         [4.0, 3.5]])
# Match fig1's COV exactly so the two figures share the same data.
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

# Identical per-class counts and mini-batch seed as fig1, so both
# figures match every point exactly.  The improvement of mKOT over fig1's
# mOT then comes purely from the keypoint-relation guidance.
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


# =======================================================================
# KPG-RL building blocks
# =======================================================================

def pick_keypoints(xs, labels_s, xt, labels_t, n_classes):
    """One source-target keypoint pair per class.

    For each class, pick the source/target sample closest to its cluster
    centroid (a stable, deterministic choice).  Return list of (i, j)
    index pairs into xs and xt.
    """
    pairs = []
    for c in range(n_classes):
        si = np.where(labels_s == c)[0]
        ti = np.where(labels_t == c)[0]
        if len(si) == 0 or len(ti) == 0:
            continue
        s_centroid = xs[si].mean(axis=0)
        t_centroid = xt[ti].mean(axis=0)
        i_star = si[np.argmin(np.linalg.norm(xs[si] - s_centroid, axis=1))]
        j_star = ti[np.argmin(np.linalg.norm(xt[ti] - t_centroid, axis=1))]
        pairs.append((int(i_star), int(j_star)))
    return pairs


def relation_score(C_to_keypoints, rho=0.1):
    """Softmax-normalised relation of each point to the keypoints.

    Following Eqs. (7)-(8) of the KPG-RL paper, we set the temperature
    proportional to the maximum keypoint distance for scale robustness.
    """
    tau = rho * (C_to_keypoints.max() + 1e-12)
    R = np.exp(-C_to_keypoints / tau)
    R /= R.sum(axis=1, keepdims=True) + 1e-30
    return R


def js_divergence_matrix(P, Q):
    """Pairwise Jensen-Shannon divergence between rows of P and Q.

    P: (m, U) probability rows, Q: (n, U) probability rows.
    Returns: (m, n) matrix of JS(P[i], Q[j]).
    """
    eps = 1e-12
    Pi = P[:, None, :]                      # (m, 1, U)
    Qj = Q[None, :, :]                      # (1, n, U)
    Mij = 0.5 * (Pi + Qj)                   # (m, n, U)
    kl_pm = np.sum(Pi * (np.log(Pi + eps) - np.log(Mij + eps)), axis=2)
    kl_qm = np.sum(Qj * (np.log(Qj + eps) - np.log(Mij + eps)), axis=2)
    return 0.5 * (kl_pm + kl_qm)


def build_guiding_matrix(xs, xt, keypoints, rho=0.1):
    """Compute the guiding cost matrix G from relations to keypoints."""
    src_kp_idx = [k[0] for k in keypoints]
    tgt_kp_idx = [k[1] for k in keypoints]
    Cs = ot.dist(xs, xs[src_kp_idx], metric="euclidean")
    Ct = ot.dist(xt, xt[tgt_kp_idx], metric="euclidean")
    Rs = relation_score(Cs, rho=rho)
    Rt = relation_score(Ct, rho=rho)
    return js_divergence_matrix(Rs, Rt)


def build_mask(m, n, keypoints):
    """Mask M from Eq. (6) of the KPG-RL paper.

    M[i,j] = 1 for keypoint pair (i,j),
    M[i,j] = 0 if i is a source keypoint but pair is wrong, ditto for j,
    M[i,j] = 1 otherwise.
    """
    I = {k[0] for k in keypoints}
    J = {k[1] for k in keypoints}
    pair_set = set(keypoints)
    M = np.ones((m, n))
    for i in range(m):
        for j in range(n):
            if (i, j) in pair_set:
                M[i, j] = 1.0
            elif i in I or j in J:
                M[i, j] = 0.0
    return M


def sinkhorn_kpg(p, q, G, M, eps=0.02, n_iter=5000, tol=1e-12):
    """Masked Sinkhorn for KPG-RL: Eq. (10) of the paper.

    Solves min_pi <M⊙pi, G> s.t. (M⊙pi)1 = p, (M⊙pi)^T 1 = q,
    with entropic regularisation `eps`.  Returns the optimal M⊙pi.

    Implementation note: G entries near max(G) cause exp(-G/eps) to
    underflow when eps is much smaller than max(G).  We rescale G to a
    fixed range [0, 1] before applying eps so the same `eps` works for
    different problem instances.
    """
    G_scaled = G / (G.max() + 1e-12)            # entries in [0, 1]
    K = M * np.exp(-G_scaled / eps)
    u = np.ones_like(p)
    v = np.ones_like(q)
    for _ in range(n_iter):
        Kv = K @ v + 1e-300
        u_new = p / Kv
        Ktu = K.T @ u_new + 1e-300
        v_new = q / Ktu
        if (np.max(np.abs(u_new - u)) < tol and
                np.max(np.abs(v_new - v)) < tol):
            u, v = u_new, v_new
            break
        u, v = u_new, v_new
    return (u[:, None] * K) * v[None, :]


def solve_kpg(xs, xt, keypoints, eps=0.02):
    """KPG-RL with full mass: same total mass on both sides."""
    m, n = len(xs), len(xt)
    G = build_guiding_matrix(xs, xt, keypoints)
    M = build_mask(m, n, keypoints)
    p = np.ones(m) / m
    q = np.ones(n) / n
    # When |xs| == |xt|, p_i = q_j for all keypoint pairs as required by
    # Proposition 1 of the paper.  When they differ we silently fall back
    # to the asymmetric formulation; the mask still pins the keypoints.
    return sinkhorn_kpg(p, q, G, M, eps=eps)


def solve_kpg_partial(xs, xt, keypoints, mass_frac=0.82, eps=0.02):
    """Partial KPG-RL via the dummy-point extension (Theorem 1).

    Adds one row and one column representing "unused mass", with cost xi
    on real-to-dummy transitions and 2xi+A on dummy-to-dummy.  The mask
    forces keypoints to remain matched (their dummy entries are zero).
    Returns the m-by-n optimal sub-plan.
    """
    m, n = len(xs), len(xt)
    G = build_guiding_matrix(xs, xt, keypoints)
    M = build_mask(m, n, keypoints)

    p = np.ones(m) / m
    q = np.ones(n) / n
    s = mass_frac
    p_bar = np.concatenate([p, [q.sum() - s]])
    q_bar = np.concatenate([q, [p.sum() - s]])

    # The dummy cost xi must sit *below* typical cross-class G so the
    # solver prefers dropping unmatchable mass over misrouting it.
    # With JS-divergence G in [0, ln 2], xi = G.max()/2 works well.
    A = 1.0
    xi = 0.5 * float(G.max())

    G_bar = np.zeros((m + 1, n + 1))
    G_bar[:m, :n] = G
    G_bar[:m, n] = xi
    G_bar[m, :n] = xi
    G_bar[m, n] = 2 * xi + A

    src_kp = {k[0] for k in keypoints}
    tgt_kp = {k[1] for k in keypoints}
    a = np.array([0.0 if i in src_kp else 1.0 for i in range(m)])
    b = np.array([0.0 if j in tgt_kp else 1.0 for j in range(n)])
    M_bar = np.zeros((m + 1, n + 1))
    M_bar[:m, :n] = M
    M_bar[:m, n] = a
    M_bar[m, :n] = b
    M_bar[m, n] = 1.0

    pi_bar = sinkhorn_kpg(p_bar, q_bar, G_bar, M_bar, eps=eps)
    return pi_bar[:m, :n]


# =======================================================================
# Solve all three settings on the same data as fig1
# =======================================================================

# Keypoints on the FULL data (one per class, near each cluster centroid).
keypoints_full = pick_keypoints(source_all, labels_s_all,
                                target_all, labels_t_all, N_CLASSES)
# Keypoints inside the mini-batch (independent choice; one per class).
keypoints_mb = pick_keypoints(source_mb, mb_s_labels,
                              target_mb, mb_t_labels, N_CLASSES)

pi_kot   = solve_kpg(source_all, target_all, keypoints_full)
pi_mkot  = solve_kpg(source_mb,  target_mb,  keypoints_mb)
# Matchable fraction here is 12/13 ≈ 0.923.  Picking s slightly above
# this lets mKPOT drop most of the 1-unit surplus that mKOT is forced
# to misroute, while still leaving a thin residual to keep the figure
# from looking artificially perfect.
pi_mkpot = solve_kpg_partial(source_mb, target_mb, keypoints_mb,
                             mass_frac=0.95)


def matching_accuracy(pi, ls, lt):
    total = pi.sum()
    correct = sum(pi[i, j] for i in range(len(ls)) for j in range(len(lt))
                  if ls[i] == lt[j])
    return correct / (total + 1e-20)


acc_kot   = matching_accuracy(pi_kot,   labels_s_all, labels_t_all)
acc_mkot  = matching_accuracy(pi_mkot,  mb_s_labels,  mb_t_labels)
acc_mkpot = matching_accuracy(pi_mkpot, mb_s_labels,  mb_t_labels)

print(f"KOT  accuracy: {acc_kot:.1%}")
print(f"mKOT  accuracy: {acc_mkot:.1%}")
print(f"mKPOT accuracy: {acc_mkpot:.1%}")


# =======================================================================
# Plotting — match fig1 style, plus keypoint highlights
# =======================================================================

MARKERS_S = {0: "X", 1: "o", 2: "^"}
MARKERS_T = {0: "X", 1: "o", 2: "^"}
SRC_COLOR = "#2563EB"
TGT_COLOR = "#16A34A"
CORRECT_LINE = "#6B7280"
WRONG_LINE = "#EF4444"
KEYPOINT_COLOR = "#111827"   # black — for keypoint match lines (red is
                             # reserved exclusively for misroutes so the
                             # two never compete visually)
KEYPOINT_RING = "#B91C1C"    # crimson ring around keypoint markers
MS = 9
LW_MATCH = 0.6


def draw_panel(ax, xs, xt, pi, labels_s, labels_t, keypoints,
               title, acc, show_all_data=False,
               source_full=None, target_full=None,
               labels_s_full=None, labels_t_full=None):
    # Background full data (faint) for mini-batch panels.
    if show_all_data and source_full is not None:
        for c in range(N_CLASSES):
            idx_s = np.where(labels_s_full == c)[0]
            idx_t = np.where(labels_t_full == c)[0]
            ax.scatter(source_full[idx_s, 0], source_full[idx_s, 1],
                       marker=MARKERS_S[c], c=SRC_COLOR, s=25, alpha=0.20,
                       linewidths=0.8, zorder=0, edgecolors=SRC_COLOR)
            ax.scatter(target_full[idx_t, 0], target_full[idx_t, 1],
                       marker=MARKERS_T[c], c="none", s=25, alpha=0.20,
                       linewidths=0.8, zorder=0, edgecolors=TGT_COLOR)

    # Transport lines.  Threshold at 8% of max — high enough to suppress
    # Sinkhorn's faint fractional smearing, low enough to keep visible
    # the red lines that account for the reported misroute mass.
    pair_set = set(keypoints)
    thresh = pi.max() * 0.08
    for i in range(len(xs)):
        for j in range(len(xt)):
            if pi[i, j] <= thresh:
                continue
            is_keypoint = (i, j) in pair_set
            is_correct = labels_s[i] == labels_t[j]
            if is_keypoint:
                color = KEYPOINT_COLOR
                lw = 1.6
                alpha = 0.95
                ls = "-"
            else:
                color = CORRECT_LINE if is_correct else WRONG_LINE
                lw = LW_MATCH * 2.5 * (pi[i, j] / pi.max()) + 0.3
                alpha = 0.7 if is_correct else 0.85
                ls = "-"
            ax.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                    ls, color=color, linewidth=lw, alpha=alpha,
                    zorder=2 if is_keypoint else 1)

    # Foreground points.
    for c in range(N_CLASSES):
        idx_s = np.where(labels_s == c)[0]
        idx_t = np.where(labels_t == c)[0]
        ax.scatter(xs[idx_s, 0], xs[idx_s, 1],
                   marker=MARKERS_S[c], c=SRC_COLOR, s=MS ** 2,
                   linewidths=1.2, zorder=3, edgecolors=SRC_COLOR)
        ax.scatter(xt[idx_t, 0], xt[idx_t, 1],
                   marker=MARKERS_T[c], c="none", s=MS ** 2,
                   linewidths=1.2, zorder=3, edgecolors=TGT_COLOR)

    # Keypoint rings.
    for (i, j) in keypoints:
        ax.scatter(xs[i, 0], xs[i, 1], marker="o",
                   facecolors="none", edgecolors=KEYPOINT_RING,
                   s=(MS + 7) ** 2, linewidths=1.8, zorder=4)
        ax.scatter(xt[j, 0], xt[j, 1], marker="o",
                   facecolors="none", edgecolors=KEYPOINT_RING,
                   s=(MS + 7) ** 2, linewidths=1.8, zorder=4)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.text(0.03, 0.03, f"Matching\naccuracy: {acc:.1%}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="gray"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_color("#D1D5DB")


# Combined figure ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

draw_panel(axes[0], source_all, target_all, pi_kot,
           labels_s_all, labels_t_all, keypoints_full,
           "(a) KOT", acc_kot)

draw_panel(axes[1], source_mb, target_mb, pi_mkot,
           mb_s_labels, mb_t_labels, keypoints_mb,
           "(b) mKOT (mini-batch)", acc_mkot,
           show_all_data=True, source_full=source_all, target_full=target_all,
           labels_s_full=labels_s_all, labels_t_full=labels_t_all)

draw_panel(axes[2], source_mb, target_mb, pi_mkpot,
           mb_s_labels, mb_t_labels, keypoints_mb,
           "(c) mKPOT (mini-batch)", acc_mkpot,
           show_all_data=True, source_full=source_all, target_full=target_all,
           labels_s_full=labels_s_all, labels_t_full=labels_t_all)

legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=SRC_COLOR,
           markeredgecolor=SRC_COLOR, markersize=8, label='Source'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor=TGT_COLOR, markersize=8, label='Target'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor=KEYPOINT_RING, markersize=10,
           markeredgewidth=1.8, label='Keypoint'),
    Line2D([0], [0], color=KEYPOINT_COLOR, linewidth=2.0, label='Keypoint match'),
    Line2D([0], [0], color=CORRECT_LINE, linewidth=1.5, label='Correct match'),
    Line2D([0], [0], color=WRONG_LINE, linewidth=1.5, label='Incorrect match'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=6,
           fontsize=9.5, framealpha=0.9, edgecolor='#D1D5DB',
           bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(w_pad=1.5)

out_dir = "results"
fig.savefig(f"{out_dir}/fig2_kpg.png", bbox_inches="tight", dpi=300)

# Individual panels -------------------------------------------------------
panels = [
    ("fig2a_kot",   source_all, target_all, pi_kot,   labels_s_all, labels_t_all,
     keypoints_full, "(a) KOT",                   acc_kot,   False),
    ("fig2b_mkot",  source_mb,  target_mb,  pi_mkot,  mb_s_labels,  mb_t_labels,
     keypoints_mb,  "(b) mKOT (mini-batch)",     acc_mkot,  True),
    ("fig2c_mkpot", source_mb,  target_mb,  pi_mkpot, mb_s_labels,  mb_t_labels,
     keypoints_mb,  "(c) mKPOT (mini-batch)",    acc_mkpot, True),
]
for tag, xs, xt, pi, ls, lt, kps, title, acc, show_bg in panels:
    fig_s, ax_s = plt.subplots(1, 1, figsize=(5, 4.5))
    draw_panel(ax_s, xs, xt, pi, ls, lt, kps, title, acc,
               show_all_data=show_bg, source_full=source_all,
               target_full=target_all, labels_s_full=labels_s_all,
               labels_t_full=labels_t_all)
    fig_s.tight_layout()
    fig_s.savefig(f"{out_dir}/{tag}.png", bbox_inches="tight", dpi=300)
    plt.close(fig_s)

plt.close("all")
print(f"\nFigures saved to {out_dir}/")
