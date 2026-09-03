"""
Colour transfer with the mini-batch transport family (Sec. V of the paper, new task).

Two groups of functions live here:

1.  The verbatim baseline transforms of the Mini-batch-OT template
    (`transform_mOT`, `transform_mUOT`, `transform_mPOT`) — kept unchanged for
    reference and cross-checking against the original pipeline.

2.  `transform_mKUOT` — ONE pipeline for the whole family, dispatched on
    (method, n_kp, alpha):

        method="OT" ,  n_kp=0            ->  m-OT      (balanced,   unguided)
        method="UOT",  n_kp=0            ->  m-UOT     (unbalanced, unguided)
        method="POT",  n_kp=0            ->  m-POT     (partial,    unguided)
        method="UOT",  n_kp=k, alpha<1   ->  m-KUOT    (unbalanced, keypoint-guided)
        method="OT"/"POT", n_kp=k        ->  m-KOT / m-KPOT (guided variants)

    The unguided members are the alpha = 1 / all-ones-mask corner of the guided
    problem (paper, Sec. IV-B), so any difference between a guided and an
    unguided run is attributable to keypoint guidance alone — same sampler,
    same solver, same reconstruction.

Keypoints (unsupervised, DeepGM protocol): a preliminary UNGUIDED plan pi_init
is solved once on a large random batch and its highest-mass entries — without
row or column reuse — define the k fixed (source colour, target colour) pairs.
The pairs occupy the first k slots of every mini-batch (Eq. 5), the mask of
Eq. (6) pins them to each other, and the JSD guiding matrix of Eq. (9) built
from relation profiles (Eq. 8) is blended into the cost (Eq. 10).

Reconstruction: the template accumulates raw barycentric updates and rescales
by the global max.  That is only meaningful when every palette entry is sampled
equally often — with keypoints in every mini-batch the k keypoint rows receive
a contribution from every sub-batch pair (paper, Sec. IV-C), so `transform_mKUOT`
uses the mass-normalised barycentric projection instead,

    colour(i) = sum_batches (T @ t)_i / sum_batches (T 1)_i ,

which for the unguided balanced case coincides with the template's output up to
a global linear rescale (row masses are exactly 1/m there).
"""

import random

import numpy as np
import ot
import tqdm

from kpg_ot import (
    build_mask,
    blend_cost,
    check_batch_feasibility,
    guiding_matrix,
    solve_kuot,
    _select_keypoint_pairs_from_plan,
)


# =======================================================================
# 1. Baseline transforms — verbatim from the Mini-batch-OT template
# =======================================================================

def transform_mOT(src, target, src_label, origin, k, m, iter=100):
    """
    Mini-batch Optimal Transport (mOT).
    Computes barycentric projection mapping source colors to target colors
    using exact Earth Mover's Distance (EMD) over mini-batches.
    """
    np.random.seed(1)
    random.seed(1)
    ot_transf = np.zeros_like(src)
    n = src.shape[0]
    for _ in tqdm.tqdm(range(iter)):
        s = np.copy(src).reshape(-1, 3).astype(float)
        t = np.array(target).reshape(-1, 3).astype(float)

        # Draw k mini-batches of size m from both source and target images
        inds1 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        inds2 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        for mi in range(k):
            for mj in range(k):
                indms = inds1[mi]
                indmt = inds2[mj]
                ms = s[indms]
                mt = t[indmt]

                # Pairwise cost matrix between sampled source and target batches
                M = ot.dist(ms, mt)

                # Solve optimal transport plan
                plan = ot.emd([], [], M, numItermax=500000)

                # Barycentric mapping: update the source colors using the target colors
                ot_transf[indms] += 1.0 / (k**2) * m * plan.dot(t[indmt])

    # Reconstruct the transformed image into the original shape
    img_ot_transf = ot_transf[src_label].reshape(origin.shape)
    img_ot_transf = img_ot_transf / np.max(img_ot_transf) * 255
    img_ot_transf = img_ot_transf.astype("uint8")
    return ot_transf, img_ot_transf


def transform_mUOT(src, target, src_label, origin, k, m, reg, tau, iter=100):
    np.random.seed(1)
    random.seed(1)
    ot_transf = np.zeros_like(src)
    n = src.shape[0]
    for _ in tqdm.tqdm(range(iter)):
        s = np.copy(src).reshape(-1, 3).astype(float)
        t = np.array(target).reshape(-1, 3).astype(float)
        inds1 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        inds2 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        for mi in range(k):
            for mj in range(k):
                indms = inds1[mi]
                indmt = inds2[mj]
                ms = s[indms]
                mt = t[indmt]
                M = ot.dist(ms, mt)
                plan = ot.unbalanced.sinkhorn_knopp_unbalanced(np.ones(m) / m, np.ones(m) / m, M, reg=reg, reg_m=tau)
                ot_transf[indms] += 1.0 / (k**2) * m * plan.dot(t[indmt])
    img_ot_transf = ot_transf[src_label].reshape(origin.shape)
    img_ot_transf = img_ot_transf / np.max(img_ot_transf) * 255
    img_ot_transf = img_ot_transf.astype("uint8")
    return ot_transf, img_ot_transf


def transform_mPOT(src, target, src_label, origin, k, m, mass, iter=100):
    np.random.seed(1)
    random.seed(1)
    ot_transf = np.zeros_like(src)
    n = src.shape[0]
    for _ in tqdm.tqdm(range(iter)):
        s = np.copy(src).reshape(-1, 3).astype(float)
        t = np.array(target).reshape(-1, 3).astype(float)
        inds1 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        inds2 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        for mi in range(k):
            for mj in range(k):
                indms = inds1[mi]
                indmt = inds2[mj]
                ms = s[indms]
                mt = t[indmt]
                M = ot.dist(ms, mt)
                plan = ot.partial.partial_wasserstein(np.ones(m) / m, np.ones(m) / m, M, m=mass)
                ot_transf[indms] += 1.0 / (k**2) * m * plan.dot(t[indmt])
    img_ot_transf = ot_transf[src_label].reshape(origin.shape)
    img_ot_transf = img_ot_transf / np.max(img_ot_transf) * 255
    img_ot_transf = img_ot_transf.astype("uint8")
    return ot_transf, img_ot_transf


# =======================================================================
# 2. m-KUOT — Sec. IV pipeline for colour transfer
# =======================================================================

def _mine_keypoints(s, t, n_kp, method, reg, tau, mass, init_m, kp_strategy, rng):
    """Mine the k fixed keypoint pairs from a preliminary unguided plan (DeepGM protocol).

    A large random batch (init_m per side) is drawn with a DEDICATED RandomState so the
    main-loop sampling sequence is untouched, the unguided problem is solved on the
    max-normalised cost, and the top-n_kp entries of the plan — no row or column reuse —
    give the (source colour, target colour) pairs.

    `kp_strategy` transposes the three target-member choices of the paper's Sec. V to the
    unlabeled setting (the source members are shared by all three, so the spread between
    strategies isolates target-keypoint quality):
      "c" (centroid analogue): the pi_init partner — the highest-confidence correspondence.
      "r" (random):            a random target palette colour.
      "f" (farthest):          among the init batch, the target colours farthest from the
                               init target batch mean — deliberately peripheral picks.
    """
    n_s, n_t = s.shape[0], t.shape[0]
    ii = rng.choice(n_s, min(init_m, n_s), replace=False)
    jj = rng.choice(n_t, min(init_m, n_t), replace=False)
    C0 = ot.dist(s[ii], t[jj])
    C0 = C0 / (C0.max() + 1e-30)
    ones_mask = np.ones_like(C0)
    eps0 = reg if (method != "UOT" or reg > 0) else 0.01
    pi_init = solve_kuot(C0, ones_mask,
                         ot_type={"OT": "balanced", "UOT": "unbalanced", "POT": "partial"}[method],
                         eps=eps0, tau=tau, mass=mass)
    I_kp, J_kp = _select_keypoint_pairs_from_plan(pi_init, n_kp)
    if len(I_kp) < n_kp:
        raise RuntimeError("keypoint mining found fewer than n_kp usable pairs")
    kp_src = ii[np.asarray(I_kp, dtype=int)]

    if kp_strategy == "c":
        kp_tgt = jj[np.asarray(J_kp, dtype=int)]
    elif kp_strategy == "r":
        kp_tgt = rng.choice(n_t, n_kp, replace=False)
    elif kp_strategy == "f":
        d_to_mean = np.linalg.norm(t[jj] - t[jj].mean(axis=0, keepdims=True), axis=1)
        kp_tgt = jj[np.argsort(-d_to_mean)[:n_kp]]
    else:
        raise ValueError(f"unknown kp_strategy {kp_strategy!r}")
    return np.asarray(kp_src, dtype=int), np.asarray(kp_tgt, dtype=int)


def transform_mKUOT(src, target, src_label, origin, k, m,
                    method="UOT", n_kp=5, alpha=0.5, rho=0.1,
                    reg=0.01, tau=1.0, mass=0.9,
                    iter=100, init_m=500, n_sink=500, tol=1e-7,
                    kp_strategy="c", seed=1, desc=None):
    """Keypoint-guided mini-batch transport for colour transfer — Eq. (11) per sub-batch.

    With n_kp=0 the mask is all-ones, the guiding matrix vanishes and alpha is forced to 1:
    the run IS the plain unguided member (m-OT / m-UOT / m-POT) of Eq. (3) on the identical
    pipeline, which is the within-regime comparison the paper prescribes.

    Returns (values, img, info): the transferred palette (n, 3), the reconstructed uint8
    image, and a dict with the mined keypoint indices/colours.
    """
    np.random.seed(seed)
    random.seed(seed)
    s = np.copy(src).reshape(-1, 3).astype(np.float64)
    t = np.asarray(target).reshape(-1, 3).astype(np.float64)
    n_s, n_t = s.shape[0], t.shape[0]

    if n_kp == 0:
        alpha = 1.0                      # no keypoints -> no guidance term to blend
        kp_src = np.empty(0, dtype=int)
        kp_tgt = np.empty(0, dtype=int)
    else:
        check_batch_feasibility(m, n_kp, strict=True)
        rng = np.random.RandomState(seed + 12345)
        kp_src, kp_tgt = _mine_keypoints(s, t, n_kp, method, reg, tau, mass,
                                         init_m, kp_strategy, rng)

    M = build_mask(m, n_kp)
    ot_type = {"OT": "balanced", "UOT": "unbalanced", "POT": "partial"}[method]
    eps = reg if (method != "UOT" or reg > 0) else 0.01
    a = np.full(m, 1.0 / m)
    b = np.full(m, 1.0 / m)

    # Sampling pools exclude the keypoints (Eq. 5: the remaining m-k points are drawn
    # from X \ keypoints), so no batch slot is duplicated.
    pool_s = np.setdiff1d(np.arange(n_s), kp_src)
    pool_t = np.setdiff1d(np.arange(n_t), kp_tgt)
    m_free = m - n_kp

    num = np.zeros((n_s, 3), dtype=np.float64)
    den = np.zeros(n_s, dtype=np.float64)
    sq = np.zeros((n_s, 3), dtype=np.float64)   # for the cross-batch dispersion

    for _ in tqdm.tqdm(range(iter), desc=desc or f"m-K{method} a={alpha}"):
        free1 = np.random.choice(pool_s, k * m_free, replace=False).reshape(k, m_free)
        free2 = np.random.choice(pool_t, k * m_free, replace=False).reshape(k, m_free)
        inds1 = [np.concatenate([kp_src, f]) for f in free1]
        inds2 = [np.concatenate([kp_tgt, f]) for f in free2]
        for mi in range(k):
            for mj in range(k):
                indms = inds1[mi]
                indmt = inds2[mj]
                ms = s[indms]
                mt = t[indmt]
                C = ot.dist(ms, mt)                       # task cost (squared Euclidean)
                G = guiding_matrix(ms, mt, n_kp, rho=rho) # Eq. (8)-(9); zero when n_kp=0
                C_tilde = blend_cost(C, G, alpha)         # Eq. (10)
                plan = solve_kuot(C_tilde, M, ot_type=ot_type, eps=eps, tau=tau,
                                  mass=mass, n_iter=n_sink, a=a, b=b, tol=tol)
                mapped = plan.dot(mt)
                rowmass = plan.sum(axis=1)
                num[indms] += mapped
                den[indms] += rowmass
                # mass-weighted second moment of the PER-BATCH barycentric image
                # x_b = mapped / rowmass:  sum_b w_b x_b^2 = mapped^2 / rowmass
                sq[indms] += mapped ** 2 / np.maximum(rowmass, 1e-30)[:, None]

    # Mass-normalised barycentric projection; palette entries that were never
    # sampled (or whose mass was entirely destroyed) keep their source colour.
    values = np.where(den[:, None] > 1e-12, num / np.maximum(den[:, None], 1e-12), s)
    values = np.clip(values, 0.0, 255.0)
    img = values[src_label].reshape(origin.shape).astype("uint8")

    # Cross-batch dispersion (the mismatching signal): the mass-weighted std of
    # a palette entry's per-batch barycentric image across all sub-batch pairs.
    # Zero would mean every mini-batch maps the colour to the same place.
    safe_den = np.maximum(den[:, None], 1e-12)
    var = np.maximum(sq / safe_den - (num / safe_den) ** 2, 0.0)
    batch_std = np.where(den > 1e-12, np.sqrt(var.sum(axis=1)), np.nan)

    info = {
        "kp_src_idx": kp_src, "kp_tgt_idx": kp_tgt,
        "kp_src_colors": s[kp_src] if n_kp else np.empty((0, 3)),
        "kp_tgt_colors": t[kp_tgt] if n_kp else np.empty((0, 3)),
        "method": method, "alpha": alpha, "n_kp": n_kp, "kp_strategy": kp_strategy,
        "batch_std": batch_std,
    }
    return values, img, info


# =======================================================================
# 3. Evaluation — palette fidelity as an exact Wasserstein-2 distance
# =======================================================================

def w2_palette_distance(values_a, weights_a, values_b, weights_b):
    """Exact squared W2 between two weighted colour palettes, colours scaled to [0, 1]^3.

    Colour transfer moves the source palette onto the target palette, so the distance of
    the TRANSFERRED palette (weighted by cluster pixel counts) to the TARGET palette is the
    natural fidelity measure: lower = the recoloured image matches the target's colour
    distribution more closely.
    """
    A = np.asarray(values_a, dtype=np.float64) / 255.0
    B = np.asarray(values_b, dtype=np.float64) / 255.0
    wa = np.asarray(weights_a, dtype=np.float64)
    wb = np.asarray(weights_b, dtype=np.float64)
    wa = wa / wa.sum()
    wb = wb / wb.sum()
    C = ot.dist(A, B)
    return float(ot.emd2(wa, wb, C, numItermax=2000000))
