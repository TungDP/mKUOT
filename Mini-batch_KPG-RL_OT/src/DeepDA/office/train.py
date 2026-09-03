import argparse
import os
import os.path as osp
import random
import lr_schedule
import network
import numpy as np
import ot
import pre_process as prep
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_list import BalancedBatchSampler, ImageList, ImageList_label
from torch.utils.data import DataLoader
from tqdm import tqdm

# =======================================================================
# m-KUOT / m-KOT / m-KPOT -- Sec. IV of the paper, inlined.
#
# Self-contained implementation of the method: mini-batch sampling with the k FIXED
# keypoint pairs in the first k slots (Eq. 5), the binary mask (Eq. 6) and its masked
# feasible set (Eq. 7), relation profiles (Eq. 8), the JSD guiding matrix (Eq. 9), the
# blended cost (Eq. 10), and the masked solvers -- for the unbalanced regime the scaling
# algorithm derived in Sec. IV-B, Eqs. (12)-(15).
#
# Convergence of the scaling iterations is governed by p = tau/(tau+eps): the closer to 1,
# the slower.  Max residual of the first-order condition Eq. (13) at eps = 0.01 --
#   Digits tau=1.0 (p=0.990): 2.2e-03 @ N=200, 2.6e-10 @ N=1000
#   VisDA  tau=0.3 (p=0.968): 3.9e-07 @ N=200, 2.8e-13 @ N=500
#   PDA    tau=0.06 (p=0.857): 4.7e-14 already at N=200
# Hence n_iter defaults to 1000.  Eq. (15) is run in the log domain (identical fixed
# point, no overflow at eps = 0.01).
#
# NOTE ON KEYPOINTS: the mask acts on rows/columns of T, so a keypoint must occupy a batch
# slot -- a class *centroid* is not a sample and cannot.  Wherever Sec. V-A-3 says
# "centroid" this code uses the class MEDOID (the sample nearest the centroid).
# =======================================================================

_TINY = 1e-30

JSD_MAX = np.log(2.0)

def build_mask(m: int, k: int) -> np.ndarray:
    """Binary mask of Eq. (6).

        M = [[ I_k ,      0      ],
             [  0  , 1_{(m-k)^2} ]]

    Row/column c <= k has a single admissible entry (c, c): keypoint x_c may send mass only
    to y_c and y_c may receive only from x_c.  The lower-right block is ALL ONES, leaving the
    m-k sampled points free to match one another.

    (The lower-right block is all-ones, not all-zeros.  With a zero block no mass could move
    between non-keypoint samples and the marginal condition would be unsatisfiable — see
    notes/OPEN-ITEMS.md in the paper repository.)
    """
    if not (0 <= k <= m):
        raise ValueError(f"need 0 <= k <= m, got k={k}, m={m}")
    M = np.zeros((m, m), dtype=np.float64)
    if k > 0:
        M[np.arange(k), np.arange(k)] = 1.0
    if m - k > 0:
        M[k:, k:] = 1.0
    return M

def check_batch_feasibility(m: int, k: int, strict: bool = True) -> None:
    """Guard the degenerate configuration k == m.

    With k == m every batch slot is a keypoint, the mask collapses to the identity, and no
    free transport remains: the method becomes a no-op that merely re-matches the fixed
    pairs.  The paper's Office-Home closed-set setting (m = 65, one keypoint per class over
    65 classes) hits exactly this case, so it CANNOT be run under the Sec. IV formulation
    without either enlarging m or using fewer keypoints.
    """
    if k > m:
        raise ValueError(f"k={k} exceeds the mini-batch size m={m}")
    if k == m:
        msg = (
            f"DEGENERATE CONFIGURATION: k == m == {m}.  Every batch slot would be a "
            f"keypoint, the mask of Eq. (6) collapses to the identity and no free transport "
            f"remains.  Increase the batch size (e.g. --batch_size {2 * m}) or reduce the "
            f"number of keypoints (e.g. --kp_per_class 1 over fewer classes)."
        )
        if strict:
            raise ValueError(msg)
        print("[mkuot] WARNING: " + msg)
    elif m - k < k:
        print(
            f"[mkuot] WARNING: only m-k={m - k} free slots for k={k} keypoints "
            f"(m={m}).  The batch is dominated by keypoints; consider a larger m."
        )

def assemble_batch_indices(kp_indices, pool_indices, m, rng, replace=False):
    """Indices of one mini-batch, keypoints first — Eq. (5).

    Returns an array of length m whose first k entries are `kp_indices` (in order, so slot c
    is keypoint c on both sides) and whose remaining m-k entries are drawn from
    `pool_indices` with the keypoints excluded.

    Parameters
    ----------
    kp_indices  : sequence of int, the k fixed keypoint indices (dataset-level).
    pool_indices: candidate indices to sample the remainder from.
    m           : mini-batch size.
    rng         : numpy Generator.
    replace     : sample the remainder with replacement (Sec. IV-A allows either).
    """
    kp = np.asarray(list(kp_indices), dtype=np.int64)
    k = kp.shape[0]
    check_batch_feasibility(m, k, strict=False)

    pool = np.setdiff1d(np.asarray(pool_indices, dtype=np.int64), kp, assume_unique=False)
    n_free = m - k
    if n_free <= 0:
        return kp[:m]
    if not replace and pool.shape[0] < n_free:
        replace = True  # pool too small to fill the batch without repetition
    rest = rng.choice(pool, size=n_free, replace=replace)
    return np.concatenate([kp, rest])

def pairwise_distance(A: np.ndarray, B: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Ground metric between every row of A and every row of B.

    The matmul is wrapped in np.errstate because some BLAS builds (notably Apple
    Accelerate under NumPy 2.x) raise spurious divide-by-zero / overflow / invalid FPE
    warnings from `A @ B.T` even for well-scaled finite inputs.  The result is asserted
    finite below, so genuine numerical trouble still surfaces.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sq = (
            np.sum(A ** 2, axis=1, keepdims=True)
            + np.sum(B ** 2, axis=1, keepdims=True).T
            - 2.0 * A @ B.T
        )
    if not np.all(np.isfinite(sq)):
        raise FloatingPointError(
            "non-finite pairwise distances: check the features for NaN/Inf "
            "(exploding activations upstream)"
        )
    np.maximum(sq, 0.0, out=sq)
    if metric == "sqeuclidean":
        return sq
    if metric == "euclidean":
        return np.sqrt(sq)
    raise ValueError(f"unknown metric {metric!r}")

def relation_profiles(feat, kp_feat, rho=0.1, metric="euclidean"):
    """Relation profiles of Eq. (8).

        r_{i,i'} = exp(-c_{i,i'} / (rho * cbar)) / sum_{i''} exp(-c_{i,i''} / (rho * cbar))

    with cbar = max_{i,i'} c_{i,i'} over the batch, so `rho` is dimensionless.  Rows are
    probability vectors over the k keypoints of the sample's own domain.

    `rho` replaces the tau_s / tau_t of the pre-2026-07-29 code.  It is NOT numerically
    comparable to them: that code used exp(-2 * normalised SQUARED distance / tau).
    """
    if kp_feat.shape[0] == 0:
        raise ValueError("no keypoints: cannot build relation profiles")
    C = pairwise_distance(feat, kp_feat, metric=metric)
    cbar = C.max()
    scale = rho * cbar if cbar > 0 else rho
    logits = -C / max(scale, _TINY)
    logits -= logits.max(axis=1, keepdims=True)          # stabilise
    e = np.exp(logits)
    return e / (e.sum(axis=1, keepdims=True) + _TINY)

def js_divergence_matrix(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Jensen-Shannon divergence (nats) between every row of P and every row of Q — Eq. (9).

    Returns a (|P|, |Q|) matrix with entries in [0, log 2].
    """
    P_e = P[:, None, :]
    Q_e = Q[None, :, :]
    Mid = 0.5 * (P_e + Q_e)
    kl1 = np.sum(P_e * (np.log(P_e + eps) - np.log(Mid + eps)), axis=-1)
    kl2 = np.sum(Q_e * (np.log(Q_e + eps) - np.log(Mid + eps)), axis=-1)
    G = 0.5 * (kl1 + kl2)
    return np.clip(G, 0.0, JSD_MAX)

def guiding_matrix(feat_x, feat_y, k, rho=0.1, metric="euclidean"):
    """Full m x m guiding matrix G of Eq. (9).

    `feat_x` / `feat_y` are the batch features with the k keypoints in the FIRST k rows
    (Eq. (5)).  Relations are computed for the sampled points (rows k..m-1) against the k
    keypoints of their own domain, exactly as Eq. (8) prescribes.

    Keypoint rows and columns of G are left at zero: the mask fixes those entries, so their
    guidance value is immaterial, and the keypoint pair (c, c) is then priced by the task
    cost alone (alpha * Cbar_{c,c}).
    """
    m = feat_x.shape[0]
    if feat_y.shape[0] != m:
        raise ValueError("source and target batches must have equal size")
    G = np.zeros((m, m), dtype=np.float64)
    if k == 0 or m - k <= 0:
        return G
    R_x = relation_profiles(feat_x[k:], feat_x[:k], rho=rho, metric=metric)
    R_y = relation_profiles(feat_y[k:], feat_y[:k], rho=rho, metric=metric)
    G[k:, k:] = js_divergence_matrix(R_x, R_y)
    return G

def blend_cost(C: np.ndarray, G: np.ndarray, alpha: float) -> np.ndarray:
    """Blended cost of Eq. (10):  C~ = alpha * Cbar + (1 - alpha) * G,  Cbar = C / max C."""
    cmax = C.max()
    Cbar = C / (cmax + _TINY) if cmax > 0 else C
    return alpha * Cbar + (1.0 - alpha) * G

def masked_uot_sinkhorn(C_tilde, M, tau, eps, n_iter=1000, a=None, b=None, tol=1e-9):
    """Masked entropic UOT of Eq. (12), solved by the scaling iterations of Eq. (15).

    Implements
        K = M (*) exp(-C~ / eps)                                     Eq. (14)
        f <- (u / (K g))^{tau / (tau + eps)}                          Eq. (15)
        g <- (u / (K^T f))^{tau / (tau + eps)}
        T  = diag(f) K diag(g)
    in the log domain for numerical stability (identical fixed point).

    The exponent tau/(tau+eps) in (0, 1) is the only difference from balanced Sinkhorn; as
    tau -> inf it tends to 1 and the classical algorithm is recovered.
    """
    m, n = C_tilde.shape
    if eps <= 0:
        raise ValueError("masked_uot_sinkhorn needs eps > 0 (the paper uses eps = 0.01)")
    a = np.full(m, 1.0 / m) if a is None else np.asarray(a, dtype=np.float64)
    b = np.full(n, 1.0 / n) if b is None else np.asarray(b, dtype=np.float64)

    # Forbidden entries get -inf log-kernel, contributing zero to every logsumexp.
    logK = np.where(M > 0, -C_tilde / eps, -np.inf)
    p = tau / (tau + eps)                      # the Eq. (15) exponent
    log_a, log_b = np.log(a + _TINY), np.log(b + _TINY)

    phi = np.zeros(m)                          # phi = eps * log f
    psi = np.zeros(n)                          # psi = eps * log g

    def _lse(Mat, axis):
        mx = np.max(Mat, axis=axis, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)          # all -inf row -> keep finite
        out = mx + np.log(np.sum(np.exp(Mat - mx), axis=axis, keepdims=True) + _TINY)
        return np.squeeze(out, axis=axis)

    for _ in range(n_iter):
        phi_prev = phi
        phi = p * eps * (log_a - _lse(logK + psi[None, :] / eps, axis=1))
        psi = p * eps * (log_b - _lse(logK + phi[:, None] / eps, axis=0))
        if np.max(np.abs(phi - phi_prev)) < tol:
            break

    logT = logK + phi[:, None] / eps + psi[None, :] / eps
    T = np.exp(logT)
    return np.where(M > 0, T, 0.0)

def _mask_penalised(C_tilde: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Cost with forbidden entries priced out of reach (for the off-the-shelf LP solvers).

    A finite penalty is enough: the masked problem is feasible, so no optimal solution ever
    pays BIG.  Used for the balanced and partial inner problems, whose solvers take a cost
    matrix but no support constraint.
    """
    BIG = 1e3 * (float(C_tilde.max()) + 1.0)
    return np.where(M > 0, C_tilde, BIG)

def solve_kuot(C_tilde, M, ot_type="unbalanced", eps=0.01, tau=1.0, mass=1.0,
               n_iter=1000, a=None, b=None, tol=1e-9):
    """KUOT on one mini-batch — Eq. (11), dispatched on the marginal constraint.

    ot_type
      "unbalanced" : the paper's setting; uses the derived scaling algorithm, Eq. (15).
      "balanced"   : masked balanced OT.  Exact: the keypoint rows/columns force
                     T_{c,c} = 1/m, and the remainder is an ordinary OT on the free block.
      "partial"    : masked partial OT with transported mass `mass`.

    Sec. IV-B notes that guidance is orthogonal to the marginal constraint, so the balanced
    and partial variants (m-KOT / m-KPOT) use the identical mask and guiding matrix.
    """
    m, n = C_tilde.shape
    a = np.full(m, 1.0 / m) if a is None else np.asarray(a, dtype=np.float64)
    b = np.full(n, 1.0 / n) if b is None else np.asarray(b, dtype=np.float64)

    if ot_type in ("unbalanced", "uot", "jumbot"):
        # The paper's setting: derived scaling algorithm, Eqs. (14)-(15).  No POT needed.
        # The iteration count matters -- see the convergence table in the module
        # docstring.  Convergence speed is set by p = tau/(tau+eps): the closer p is to 1,
        # the slower.  The `tol` early-exit means a generous n_iter costs nothing on the
        # easy configurations.
        return masked_uot_sinkhorn(C_tilde, M, tau=tau, eps=eps, n_iter=n_iter,
                                   a=a, b=b, tol=tol)

    import ot  # noqa: E402  (lazy: only the balanced / partial branches need POT)

    if ot_type in ("balanced", "ot", "jdot"):
        C_pen = _mask_penalised(C_tilde, M)
        T = ot.emd(a, b, C_pen) if eps == 0 else ot.sinkhorn(a, b, C_pen, reg=eps)
        return np.where(M > 0, T, 0.0)

    if ot_type in ("partial", "pot", "jpmbot"):
        C_pen = _mask_penalised(C_tilde, M)
        if eps == 0:
            T = ot.partial.partial_wasserstein(a, b, C_pen, m=mass)
        else:
            T = ot.partial.entropic_partial_wasserstein(a, b, C_pen, m=mass, reg=eps)
            if np.any(np.isnan(T)):        # entropic POT can diverge for tiny mass
                T = ot.partial.partial_wasserstein(a, b, C_pen, m=mass)
        return np.where(M > 0, T, 0.0)

    raise ValueError(f"unknown ot_type {ot_type!r}")

def class_medoid_index(feat: np.ndarray, indices: np.ndarray) -> int:
    """Index (into `indices`) of the sample closest to the centroid of feat[indices].

    The medoid is used wherever the paper's Sec. V-A-3 says "centroid": the mask needs the
    keypoint to BE a sample, and the medoid is the sample-valued stand-in for the centroid.
    """
    sub = feat[indices]
    centroid = sub.mean(axis=0, keepdims=True)
    d = np.linalg.norm(sub - centroid, axis=1)
    return int(indices[int(np.argmin(d))])

def select_keypoint_pairs(
    feat_s, labels_s, feat_t, labels_t, classes,
    strategy="centroid", kp_per_class=1, rng=None,
):
    """Choose the FIXED keypoint pairs of Eq. (4), one (or `kp_per_class`) per class.

    Source side is always the class medoid (source labels are available by definition).
    Target side follows `strategy`, which is the paper's Sec. V-A-3 ablation, restated for
    sample-valued keypoints:

      "centroid" — medoid of the true target class (needs ground-truth target labels).
                   ORACLE / upper bound, not a deployable method.
      "random"   — a random target sample of the same pseudo-class.  PRACTICAL variant.
      "farthest" — the pseudo-class sample farthest from its class centroid.  Adversarial
                   lower bound.

    Returns (src_idx, tgt_idx, kept_classes).  A class is skipped when either domain has no
    sample for it, so len(src_idx) is the effective k and may be < len(classes).
    """
    rng = np.random.default_rng() if rng is None else rng
    src_idx, tgt_idx, kept = [], [], []

    for c in classes:
        s_pool = np.where(labels_s == c)[0]
        t_pool = np.where(labels_t == c)[0]
        if s_pool.size == 0 or t_pool.size == 0:
            continue                                    # no pair possible for this class

        n_take = min(kp_per_class, s_pool.size, t_pool.size)

        for r in range(n_take):
            if r == 0:
                s_pick = class_medoid_index(feat_s, s_pool)
            else:
                s_pick = int(rng.choice(s_pool))

            if strategy == "centroid":
                t_pick = class_medoid_index(feat_t, t_pool)
            elif strategy == "random":
                t_pick = int(rng.choice(t_pool))
            elif strategy == "farthest":
                sub = feat_t[t_pool]
                cen = sub.mean(axis=0, keepdims=True)
                t_pick = int(t_pool[int(np.argmax(np.linalg.norm(sub - cen, axis=1)))])
            else:
                raise ValueError(f"unknown keypoint strategy {strategy!r}")

            if s_pick in src_idx or t_pick in tgt_idx:
                continue                                # keep pairs disjoint
            src_idx.append(s_pick)
            tgt_idx.append(t_pick)
            kept.append(int(c))

    return np.asarray(src_idx, dtype=np.int64), np.asarray(tgt_idx, dtype=np.int64), kept

@torch.no_grad()
def extract_features(forward_fn, dataset, indices, device="cuda", batch_size=64,
                     want_logits=False):
    """Encode `dataset[i]` for i in `indices`.

    `forward_fn(x)` must return either the feature tensor, or a (feature, logit) tuple —
    both driver conventions in this repo are accepted.

    Returns `feat` (N, d) float64 numpy, and if `want_logits` also `logit` (N, C).
    """
    feats, logits = [], []
    idx = np.asarray(indices, dtype=np.int64)
    for s in range(0, idx.shape[0], batch_size):
        chunk = idx[s: s + batch_size]
        xs = torch.stack([dataset[int(i)][0] for i in chunk]).to(device)
        out = forward_fn(xs)
        f, lg = out if isinstance(out, (tuple, list)) else (out, None)
        feats.append(f.detach().float().cpu().numpy())
        if want_logits and lg is not None:
            logits.append(lg.detach().float().cpu().numpy())
    feat = np.concatenate(feats, axis=0).astype(np.float64)
    if want_logits:
        lg = np.concatenate(logits, axis=0).astype(np.float64) if logits else None
        return feat, lg
    return feat

class KeypointBank:
    """Holds the k fixed keypoint pairs and builds keypoint-first mini-batches.

    Sec. IV-A requires the pairs to be fixed while the rest of the batch is resampled.  We
    therefore store dataset INDICES, chosen once by `select`, and reuse them in every batch.
    The keypoints' embeddings still move as the encoder trains — the pairs are fixed, not
    their coordinates.

    `strategy` is the Sec. V-A-3 ablation, restated for sample-valued keypoints (the mask of
    Eq. (6) acts on batch rows/columns, so a keypoint must be an actual sample; a class
    centroid is not one, hence the medoid):

        centroid  target medoid of the TRUE class      -> oracle, needs GT target labels
        random    random sample of the pseudo-class    -> practical / deployable
        farthest  pseudo-class sample farthest from its centroid -> adversarial lower bound
    """

    def __init__(self, strategy="random", kp_per_class=1, rho=0.1,
                 metric="euclidean", seed=0):
        self.strategy = strategy
        self.kp_per_class = kp_per_class
        self.rho = rho
        self.metric = metric
        self.rng = np.random.default_rng(seed)
        self.src_idx = np.zeros(0, dtype=np.int64)
        self.tgt_idx = np.zeros(0, dtype=np.int64)
        self.classes = []

    # ----- selection ---------------------------------------------------

    @property
    def k(self) -> int:
        return int(self.src_idx.shape[0])

    def is_ready(self) -> bool:
        return self.k > 0

    def select(self, feat_s, labels_s, feat_t, labels_t, classes,
               pool_s=None, pool_t=None):
        """Pick the fixed pairs from precomputed features / labels.

        `labels_t` must be ground-truth target labels for the `centroid` (oracle) strategy
        and classifier pseudo-labels otherwise.  `pool_s` / `pool_t` map row positions in
        `feat_s` / `feat_t` back to dataset indices when features were computed on a subset.
        """
        s_loc, t_loc, kept = select_keypoint_pairs(
            feat_s, labels_s, feat_t, labels_t, classes,
            strategy=self.strategy, kp_per_class=self.kp_per_class, rng=self.rng,
        )
        self.src_idx = s_loc if pool_s is None else np.asarray(pool_s, dtype=np.int64)[s_loc]
        self.tgt_idx = t_loc if pool_t is None else np.asarray(pool_t, dtype=np.int64)[t_loc]
        self.classes = kept
        return self.k

    def describe(self) -> str:
        return ("KeypointBank(strategy={}, k={}, kp_per_class={}, rho={}, classes={})"
                .format(self.strategy, self.k, self.kp_per_class, self.rho,
                        len(self.classes)))

    # ----- batching ----------------------------------------------------

    def batch_indices(self, m, n_source, n_target, replace=False):
        """Dataset indices for one mini-batch per domain, keypoints in the first k slots."""
        check_batch_feasibility(m, self.k, strict=True)
        si = assemble_batch_indices(
            self.src_idx, np.arange(n_source), m, self.rng, replace=replace)
        ti = assemble_batch_indices(
            self.tgt_idx, np.arange(n_target), m, self.rng, replace=replace)
        return si, ti

    def load_batch(self, dset_s, dset_t, m, replace=False, device="cuda"):
        """Materialise one keypoint-first mini-batch as tensors.

        Returns (xs, ys, xt, yt) with the k keypoint pairs aligned at positions 0..k-1, so
        slot c on the source side corresponds to slot c on the target side exactly as
        Eq. (5) requires.
        """
        si, ti = self.batch_indices(m, len(dset_s), len(dset_t), replace=replace)
        xs = torch.stack([dset_s[int(i)][0] for i in si]).to(device)
        ys = torch.tensor([int(dset_s[int(i)][1]) for i in si], dtype=torch.long, device=device)
        xt = torch.stack([dset_t[int(j)][0] for j in ti]).to(device)
        yt = torch.tensor([int(dset_t[int(j)][1]) for j in ti], dtype=torch.long)
        return xs, ys, xt, yt

def inject_keypoints(bank, xs, ys, xt, yt, dset_s, dset_t, device="cuda"):
    """Overwrite the first k slots of an already-fetched batch with the fixed keypoints.

    The drivers for digits / PartialDA / DeepGM obtain their mini-batches from a DataLoader.
    Rather than rewriting those data pipelines, we replace the first k slots of the fetched
    batch by the k fixed keypoint pairs.  The result is exactly the batch Eq. (5) specifies:
    keypoints at positions 0..k-1 (aligned across domains) and m-k random samples after
    them — the random samples that occupied those slots are simply dropped.

    Returns new (xs, ys, xt, yt) tensors; the originals are left untouched.
    """
    k = bank.k
    if k == 0:
        return xs, ys, xt, yt
    m = xs.shape[0]
    check_batch_feasibility(m, k, strict=True)

    kp_xs = torch.stack([dset_s[int(i)][0] for i in bank.src_idx]).to(xs.device, xs.dtype)
    kp_ys = torch.tensor([int(dset_s[int(i)][1]) for i in bank.src_idx],
                         dtype=ys.dtype, device=ys.device)
    kp_xt = torch.stack([dset_t[int(j)][0] for j in bank.tgt_idx]).to(xt.device, xt.dtype)
    kp_yt = torch.tensor([int(dset_t[int(j)][1]) for j in bank.tgt_idx],
                         dtype=yt.dtype, device=yt.device)

    xs, ys, xt, yt = xs.clone(), ys.clone(), xt.clone(), yt.clone()
    xs[:k], ys[:k] = kp_xs, kp_ys
    xt[:k], yt[:k] = kp_xt, kp_yt
    return xs, ys, xt, yt

_KP_CACHE = {"iter": -10**9, "data": None}

def inject_keypoints_cached(bank, xs, ys, xt, yt, dset_s, dset_t, it, refresh=50):
    """`inject_keypoints` with the keypoint tensors CACHED.

    The k pairs are fixed for the whole run, so re-loading and re-augmenting all 2k
    keypoint images from disk in the main process every iteration (the `load_batch`
    path) only re-rolls their random crops -- at ~35 ms/image, single-threaded, that
    is the dominant per-iteration cost at k = 65.  Caching the materialised tensors
    and refreshing them every `refresh` iterations keeps augmentation diversity at
    1/refresh of the cost.  Enabled by MKUOT_FAST_KP=1 (ablation scripts)."""
    if _KP_CACHE["data"] is None or it - _KP_CACHE["iter"] >= refresh:
        kp_xs = torch.stack([dset_s[int(i)][0] for i in bank.src_idx])
        kp_ys = torch.tensor([int(dset_s[int(i)][1]) for i in bank.src_idx], dtype=torch.long)
        kp_xt = torch.stack([dset_t[int(j)][0] for j in bank.tgt_idx])
        kp_yt = torch.tensor([int(dset_t[int(j)][1]) for j in bank.tgt_idx], dtype=torch.long)
        _KP_CACHE["iter"] = it
        _KP_CACHE["data"] = (kp_xs, kp_ys, kp_xt, kp_yt)
    kp_xs, kp_ys, kp_xt, kp_yt = _KP_CACHE["data"]
    kk = bank.k
    check_batch_feasibility(xs.shape[0], kk, strict=True)
    xs, ys, xt, yt = xs.clone(), ys.clone(), xt.clone(), yt.clone()
    xs[:kk], ys[:kk] = kp_xs.to(xs.device, xs.dtype), kp_ys.to(ys.device, ys.dtype)
    xt[:kk], yt[:kk] = kp_xt.to(xt.device, xt.dtype), kp_yt.to(yt.device, yt.dtype)
    return xs, ys, xt, yt

def guided_plan(cost, feat_s, feat_t, k, alpha=0.5, rho=0.1, metric="euclidean",
                ot_type="unbalanced", eps=0.01, tau=1.0, mass=1.0, n_iter=200,
                return_numpy=False, use_mask=True):
    """Solve KUOT on one mini-batch and return the plan.

    `cost` is the task cost C (torch tensor or numpy array) with the keypoints in the first
    k rows/columns.  `feat_s` / `feat_t` are the matching batch features.

    Steps (Alg. 1): build the mask -> relation profiles + guiding matrix -> blend -> masked
    solve.  With k == 0 this degrades to the plain unguided masked-free problem.
    """
    is_torch = torch.is_tensor(cost)
    C_np = cost.detach().cpu().numpy().astype(np.float64) if is_torch else np.asarray(cost, dtype=np.float64)

    m = C_np.shape[0]
    # use_mask=False is the A1 "mask-off" arm: the keypoints still occupy the first k
    # slots of the batch and still define the relation profiles (so the guiding matrix
    # is unchanged), but Eq. (6) is replaced by an all-ones matrix -- nothing is pinned.
    # The vertical gap to the mask-on curve at matched alpha is the mask's contribution.
    M = build_mask(m, k) if use_mask else np.ones((m, m), dtype=np.float64)
    G = guiding_matrix(feat_s, feat_t, k, rho=rho, metric=metric)
    C_tilde = blend_cost(C_np, G, alpha)
    T = solve_kuot(C_tilde, M, ot_type=ot_type, eps=eps, tau=tau, mass=mass,
                        n_iter=n_iter)

    if return_numpy or not is_torch:
        return T
    return torch.from_numpy(T).float().to(cost.device)


OFFICE_HOME_LIST_PREFIX = "/data/office-home/images/"
OFFICE31_LIST_PREFIX = "/data/office/domain_adaptation_images/"
VISDA_LIST_PREFIX = "/data/visda-2017/"

# VisDA-2017 class names in label order 0..11 (the standard results-table order;
# the 5th class is "horse", which some papers' tables misprint as "house").
VISDA_CLASSES = [
    "plane", "bcycl", "bus", "car", "horse", "knife",
    "mcycl", "person", "plant", "sktbrd", "train", "truck",
]


# -----------------------------------------------------------------------
# Data-loading helpers
# -----------------------------------------------------------------------

def build_list_path_candidates(raw_path, list_path):
    base_dir = osp.dirname(osp.abspath(list_path))
    project_dir = osp.dirname(osp.abspath(__file__))
    candidates = []

    # --- Office-Home ---------------------------------------------------
    office_home_root = os.environ.get("OFFICE_HOME_IMAGES_ROOT")
    if office_home_root:
        office_home_root = osp.abspath(osp.expanduser(office_home_root))
        if raw_path.startswith(OFFICE_HOME_LIST_PREFIX):
            candidates.append(
                osp.join(office_home_root, raw_path[len(OFFICE_HOME_LIST_PREFIX):])
            )

    if raw_path.startswith(OFFICE_HOME_LIST_PREFIX):
        candidates.append(
            osp.join(base_dir, "images", raw_path[len(OFFICE_HOME_LIST_PREFIX):])
        )

    # --- Office-31 -----------------------------------------------------
    # The list files were generated with an extra "images/" subdirectory
    # (i.e. {domain}/images/{class}/...) that is absent in the local copy
    # ({domain}/{class}/...).  Strip it when building the candidate path.
    office31_root = os.environ.get("OFFICE31_IMAGES_ROOT")
    if office31_root and raw_path.startswith(OFFICE31_LIST_PREFIX):
        office31_root = osp.abspath(osp.expanduser(office31_root))
        suffix = raw_path[len(OFFICE31_LIST_PREFIX):]   # {domain}/images/{class}/...
        parts = suffix.split("/", 2)
        if len(parts) == 3 and parts[1] == "images":
            suffix = parts[0] + "/" + parts[2]          # {domain}/{class}/...
        candidates.append(osp.join(office31_root, suffix))

    # --- VisDA-2017 ----------------------------------------------------
    # List paths look like "/data/visda-2017/{train,validation}/<class>/<img>".
    # Resolve them under $VISDA_IMAGES_ROOT (which should point at the dir that
    # contains the train/ and validation/ image folders), or under the list
    # file's own directory.
    visda_root = os.environ.get("VISDA_IMAGES_ROOT")
    if raw_path.startswith(VISDA_LIST_PREFIX):
        suffix = raw_path[len(VISDA_LIST_PREFIX):]      # {train,validation}/<class>/<img>
        if visda_root:
            visda_root = osp.abspath(osp.expanduser(visda_root))
            candidates.append(osp.join(visda_root, suffix))
        candidates.append(osp.join(base_dir, suffix))

    # --- generic fallbacks ---------------------------------------------
    candidates.extend(
        [
            raw_path,
            "." + raw_path if raw_path.startswith("/") else raw_path,
            osp.join(project_dir, raw_path.lstrip("/")),
            osp.join(base_dir, raw_path),
            osp.join(base_dir, raw_path.lstrip("./")),
            osp.join(".", raw_path.lstrip("/")),
        ]
    )
    return candidates


def load_dataset_list(list_path):
    with open(list_path) as list_file:
        lines = list_file.readlines()
    resolved_list = []

    for line in lines:
        fields = line.strip().split()
        if not fields:
            continue

        raw_path = fields[0]
        resolved_path = None
        for candidate in build_list_path_candidates(raw_path, list_path):
            if osp.exists(candidate):
                resolved_path = candidate
                break

        if resolved_path is None:
            raise FileNotFoundError(
                f"Image file not found from list '{list_path}': '{raw_path}'. "
                "Please place/extract Office-Home images under './data/office-home/images/', "
                "set OFFICE_HOME_IMAGES_ROOT to your dataset directory, or update the list paths."
            )

        resolved_list.append(" ".join([resolved_path] + fields[1:]) + "\n")

    return resolved_list


def image_classification_test(loader, model, test_10crop=True):
    all_output = []
    all_label = []
    dataset = loader["test"]

    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(dataset[i]) for i in range(10)]
            for _ in tqdm(range(len(dataset[0]))):
                data = [next(iter_test[j]) for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    predict_out = nn.Softmax(dim=1)(predict_out)
                    outputs.append(predict_out)
                outputs = sum(outputs) / 10
                all_output.append(outputs.float().cpu())
                all_label.append(labels.float())
        else:
            iter_test = iter(dataset)
            for _ in tqdm(range(len(dataset))):
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                _, outputs = model(inputs)
                outputs = nn.Softmax(dim=1)(outputs)
                all_output.append(outputs.float().cpu())
                all_label.append(labels.float())

    all_output = torch.cat(all_output, 0)
    all_label = torch.cat(all_label, 0)
    _, predict = torch.max(all_output, 1)
    predict = torch.squeeze(predict).float()

    # Overall accuracy (the metric for office / office-home).
    overall = torch.sum(predict == all_label).item() / float(all_label.size()[0])

    # Per-class accuracy + mean-per-class (the standard VisDA-2017 metric, since
    # VisDA is heavily class-imbalanced).  Returned for every dataset; the caller
    # decides which scalar to select on.
    n_class = all_output.size(1)
    per_class = np.full(n_class, np.nan, dtype=np.float64)
    for c in range(n_class):
        mask = (all_label == c)
        n_c = int(mask.sum().item())
        if n_c > 0:
            per_class[c] = torch.sum(predict[mask] == c).item() / float(n_c)
    mean_per_class = float(np.nanmean(per_class))
    return mean_per_class, per_class, overall


# -----------------------------------------------------------------------
# Unguided OT solvers -- the m-OT / m-UOT / m-POT baselines.
# The guided path is the Sec. IV block above (solve_kuot / solve_kuot_paper).
# -----------------------------------------------------------------------

def solve_ot(a, b, M_cpu, ot_type, epsilon, tau, mass):
    """Standard OT solver (balanced / unbalanced / partial).

    This is the original Mini-batch-OT logic, extracted into a helper
    so both standard and KPG-guided paths share the same dispatch.
    """
    if ot_type == "balanced":
        if epsilon == 0:
            return ot.emd(a, b, M_cpu)
        else:
            return ot.sinkhorn(a, b, M_cpu, epsilon)
    elif ot_type == "unbalanced":
        return ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M_cpu, epsilon, tau)
    elif ot_type == "partial":
        if epsilon == 0:
            return ot.partial.partial_wasserstein(a, b, M_cpu, mass)
        else:
            return ot.partial.entropic_partial_wasserstein(a, b, M_cpu, m=mass, reg=epsilon)


def build_keypoint_bank(config, base_network, dsets, class_num, device="cuda"):
    """Select the FIXED keypoint pairs of Eq. (4) once, before training.

    Source labels are ground truth; target labels are the current classifier's pseudo-labels
    except for the `centroid` oracle strategy, which consumes ground-truth target labels.
    Features come from the ImageNet-pretrained backbone at this point, so the pairs are
    chosen from the initial representation and then held fixed (Sec. IV-A).
    """
    args = config["args"]
    bank = KeypointBank(
        strategy=config["kp_strategy"],
        kp_per_class=config["kp_per_class"],
        rho=config["rho"],
        metric=config["kp_metric"],
        seed=args.seed,
    )

    n_probe = config["kp_probe"]
    rng = np.random.default_rng(args.seed)
    pool_s = np.sort(rng.choice(len(dsets["source"]),
                                size=min(n_probe, len(dsets["source"])), replace=False))
    pool_t = np.sort(rng.choice(len(dsets["target"]),
                                size=min(n_probe, len(dsets["target"])), replace=False))

    base_network.eval()
    feat_s = extract_features(base_network, dsets["source"], pool_s, device=device)
    feat_t, logit_t = extract_features(
        base_network, dsets["target"], pool_t, device=device, want_logits=True)
    base_network.train()

    lab_s = np.array([int(dsets["source"][int(i)][1]) for i in pool_s])
    if config["kp_strategy"] == "centroid":
        # Oracle: ground-truth target labels.
        lab_t = np.array([int(dsets["target"][int(j)][1]) for j in pool_t])
    else:
        lab_t = logit_t.argmax(axis=1)

    n_cls = config.get("kp_n_classes") or class_num
    k = bank.select(feat_s, lab_s, feat_t, lab_t, range(n_cls),
                    pool_s=pool_s, pool_t=pool_t)
    print("[mkuot] " + bank.describe())
    if k == 0:
        raise RuntimeError(
            "keypoint selection produced 0 pairs — no class had a sample in both domains "
            "within the probe subset; raise --kp_probe."
        )
    # With --no_mask nothing is pinned, so the k == m degeneracy the guard protects
    # against cannot arise; warn instead of raising.
    check_batch_feasibility(config["data"]["source"]["batch_size"], k,
                            strict=not config["no_mask"])
    return bank


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train(config):
    # ---- pre-processing ------------------------------------------------
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]["params"])
    prep_dict["target"] = prep.image_train(**config["prep"]["params"])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]["params"])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]["params"])

    # ---- data loaders --------------------------------------------------
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs  = data_config["test"]["batch_size"]

    source_list = load_dataset_list(data_config["source"]["list_path"])
    target_list = load_dataset_list(data_config["target"]["list_path"])

    dsets["source"] = ImageList(source_list, transform=prep_dict["source"])
    if config["args"].stratify_source:
        source_labels = torch.zeros((len(dsets["source"])))
        for i, data in tqdm(enumerate(source_list)):
            source_labels[i] = int(data.split()[1])
        source_sampler = BalancedBatchSampler(source_labels, batch_size=train_bs)
        dset_loaders["source"] = DataLoader(
            dsets["source"],
            batch_sampler=source_sampler,
            num_workers=config["args"].num_worker,
        )
    else:
        dset_loaders["source"] = DataLoader(
            dsets["source"],
            batch_size=train_bs,
            shuffle=True,
            num_workers=config["args"].num_worker,
            drop_last=True,
        )

    dsets["target"] = ImageList(target_list, transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(
        dsets["target"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=config["args"].num_worker,
        drop_last=True,
    )
    print("source dataset len:", len(dsets["source"]))
    print("target dataset len:", len(dsets["target"]))

    if prep_config["test_10crop"]:
        for i in range(10):
            test_list = load_dataset_list(data_config["test"]["list_path"])
            dsets["test"] = [ImageList(test_list, transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [
                DataLoader(dset, batch_size=test_bs, shuffle=False, num_workers=config["args"].num_worker)
                for dset in dsets["test"]
            ]
    else:
        test_list = load_dataset_list(data_config["test"]["list_path"])
        dsets["test"] = ImageList(test_list, transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(
            dsets["test"],
            batch_size=test_bs,
            shuffle=False,
            num_workers=config["args"].num_worker,
        )

    dsets["target_label"] = ImageList_label(target_list, transform=prep_dict["target"])
    dset_loaders["target_label"] = DataLoader(
        dsets["target_label"],
        batch_size=test_bs,
        shuffle=False,
        num_workers=config["args"].num_worker,
        drop_last=False,
    )

    class_num = config["network"]["params"]["class_num"]

    # ---- base network --------------------------------------------------
    net_config  = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    if config["restore_path"]:
        checkpoint = torch.load(osp.join(config["restore_path"], "best_model.pth"))
        checkpoint = checkpoint["base_network"]
        ckp = {}
        for k, v in checkpoint.items():
            ckp[k.split("module.")[-1] if "module" in k else k] = v
        base_network.load_state_dict(ckp)
        log_str = "successfully restore from {}".format(
            osp.join(config["restore_path"], "best_model.pth")
        )
        config["out_file"].write(log_str + "\n")
        config["out_file"].flush()
        print(log_str)

    parameter_list = base_network.get_parameters()

    # ---- optimizer -----------------------------------------------------
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler   = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config["gpu"].split(",")
    if len(gpus) > 1:
        base_network = nn.DataParallel(
            base_network, device_ids=[int(i) for i in range(len(gpus))]
        )

    # ---- training config -----------------------------------------------
    use_bomb  = config["use_bomb"]
    use_kpg   = config["use_kpg"]
    ot_type   = config["ot_type"]
    k         = config["k"]
    eta1      = config["eta1"]
    eta2      = config["eta2"]
    epsilon   = config["epsilon"]
    be        = config["be"]
    tau       = config["tau"]
    mass      = config["mass"]
    alpha     = config["alpha"]
    kpg_rng   = np.random.default_rng(config["args"].seed)

    # ---- Sec. IV formulation -------------------------------------------
    kp_bank = None
    if use_kpg:
        if use_bomb:
            raise NotImplementedError(
                "--use_bomb cannot be combined with --use_kpg: the hierarchical "
                "batch-of-mini-batches scheme is outside the Sec. IV formulation, whose "
                "estimator is the uniform average of Eq. (8).  Drop one of the two."
            )
        kp_bank = build_keypoint_bank(config, base_network, dsets, class_num)

    best_step = 0
    best_acc  = 0.0                                  # the selection metric
    best_per_class = np.zeros(class_num)            # VisDA per-class breakdown
    best_overall = 0.0
    is_visda = (config.get("dataset") == "visda")   # VisDA -> mean-per-class metric
    iter_source = iter(dset_loaders["source"])
    iter_target = iter(dset_loaders["target"])

    # Optional mixed precision (env MKUOT_AMP=1): halves activation memory so the
    # m=128 ablation batches fit on a single 16 GB GPU.  Off by default -- the
    # main-table runs remain full fp32.  Only the standard averaging path uses it
    # (the BoMb path is never run with keypoint guidance).
    use_amp = os.environ.get("MKUOT_AMP", "0") == "1"
    # MKUOT_FAST_KP=1: batches come from the multi-worker DataLoaders (stratified
    # source sampler) with the fixed keypoints injected from a cache, instead of
    # load_batch's single-threaded 2m-image materialisation per iteration.
    use_fast_kp = os.environ.get("MKUOT_FAST_KP", "0") == "1"
    if use_fast_kp:
        print("[mkuot] fast keypoint batching enabled via MKUOT_FAST_KP=1")
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("[mkuot] mixed precision enabled via MKUOT_AMP=1")

    for id_iter in tqdm(range(config["num_iterations"]), total=config["num_iterations"]):

        # ---- evaluation ------------------------------------------------
        if id_iter % config["test_interval"] == config["test_interval"] - 1:
            base_network.eval()
            mean_pc, per_class, overall = image_classification_test(
                dset_loaders, base_network, test_10crop=prep_config["test_10crop"]
            )
            # VisDA selects on mean-per-class; office / office-home on overall.
            temp_acc = mean_pc if is_visda else overall
            temp_model = base_network
            if temp_acc > best_acc:
                best_step      = id_iter
                best_acc       = temp_acc
                best_per_class = per_class
                best_overall   = overall
                best_model     = temp_model
                checkpoint = {"base_network": best_model.state_dict()}
                torch.save(checkpoint, osp.join(config["output_path"], "best_model.pth"))
                print("\n##########     save the best model.    #############\n")
            if is_visda:
                pc_str = "  ".join(
                    "{}={:.1f}".format(nm, 100.0 * v)
                    for nm, v in zip(VISDA_CLASSES, per_class)
                )
                log_str = (
                    "iter: {:05d}, mean_per_class: {:.5f}, overall(All): {:.5f}\n"
                    "    per_class: {}".format(id_iter, mean_pc, overall, pc_str)
                )
            else:
                log_str = "iter: {:05d}, precision: {:.5f}".format(id_iter, temp_acc)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)

        if id_iter >= config["stop_step"]:
            # train.py-native one-liner (kept so existing parsers still work).
            log_str = "method {}, iter: {:05d}, precision: {:.5f}".format(
                config["output_path"], best_step, best_acc
            )
            config["final_log"].write(log_str + "\n")
            if is_visda:
                # Paper-style per-class table for the best checkpoint:
                #   All  = overall accuracy on all 12 classes (cross-method metric)
                #   <12> = per-class precision    Avg = mean of the 12 per-class
                header = "| All | " + " | ".join(VISDA_CLASSES) + " | Avg |"
                sep    = "|" + "|".join(["-----"] * (len(VISDA_CLASSES) + 2)) + "|"
                cells  = " | ".join("{:.1f}".format(100.0 * v) for v in best_per_class)
                row    = "| {:.1f} | {} | {:.1f} |".format(
                    100.0 * best_overall, cells, 100.0 * best_acc
                )
                config["final_log"].write(header + "\n")
                config["final_log"].write(sep + "\n")
                config["final_log"].write(row + "\n")
                # Machine-parseable single line for the shell to build a
                # cumulative Run | Method | All | <12> | Avg table.
                parts = ["All={:.2f}".format(100.0 * best_overall)]
                parts += ["{}={:.2f}".format(nm, 100.0 * v)
                          for nm, v in zip(VISDA_CLASSES, best_per_class)]
                parts += ["Avg={:.2f}".format(100.0 * best_acc)]
                config["final_log"].write("RESULT " + " ".join(parts) + "\n")
            config["final_log"].flush()
            break

        # ---- sample k mini-batches -------------------------------------
        base_network.train()
        # yt_mb_all only used when kp_strategy == "centroid" (oracle)
        xs_mb_all, ys_mb_all, xt_mb_all, yt_mb_all = [], [], [], []

        if kp_bank is not None and use_fast_kp:
            # Fast path (Eq. (5) semantics preserved): fetch a normal loader batch
            # (multi-worker, stratified source sampler) and overwrite its first k
            # slots with the cached fixed keypoints -- the samples that occupied
            # those slots are dropped, exactly as in `inject_keypoints`.
            for _ in range(k):
                try:
                    xs_mb, ys_mb = next(iter_source)
                    xt_mb, yt_mb = next(iter_target)
                except StopIteration:
                    iter_source  = iter(dset_loaders["source"])
                    iter_target  = iter(dset_loaders["target"])
                    xs_mb, ys_mb = next(iter_source)
                    xt_mb, yt_mb = next(iter_target)
                xs_mb, ys_mb, xt_mb, yt_mb = inject_keypoints_cached(
                    kp_bank, xs_mb, ys_mb, xt_mb, yt_mb,
                    dsets["source"], dsets["target"], id_iter,
                )
                xs_mb_all.append(xs_mb)
                ys_mb_all.append(ys_mb)
                xt_mb_all.append(xt_mb)
                yt_mb_all.append(yt_mb)
        elif kp_bank is not None:
            # Sec. IV-A: the k fixed keypoint pairs occupy the first k slots of every
            # mini-batch (Eq. (5)); only the remaining m-k points are resampled.
            m_batch = data_config["source"]["batch_size"]
            for _ in range(k):
                xs_mb, ys_mb, xt_mb, yt_mb = kp_bank.load_batch(
                    dsets["source"], dsets["target"], m_batch,
                    replace=config["kp_replace"], device="cuda",
                )
                xs_mb_all.append(xs_mb.cpu())
                ys_mb_all.append(ys_mb.cpu())
                xt_mb_all.append(xt_mb.cpu())
                yt_mb_all.append(yt_mb)
        else:
            for _ in range(k):
                try:
                    xs_mb, ys_mb = next(iter_source)
                    xt_mb, yt_mb = next(iter_target)
                except StopIteration:
                    iter_source  = iter(dset_loaders["source"])
                    iter_target  = iter(dset_loaders["target"])
                    xs_mb, ys_mb = next(iter_source)
                    xt_mb, yt_mb = next(iter_target)
                xs_mb_all.append(xs_mb)
                ys_mb_all.append(ys_mb)
                yt_mb_all.append(yt_mb)
                xt_mb_all.append(xt_mb)

        list_transfer_loss = []

        # ================================================================
        # BoMb path: hierarchical batch-of-mini-batches OT
        # ================================================================
        if use_bomb:

            # -- forward pass (no grad): solve all k×k OT problems ------
            with torch.no_grad():
                for i in range(k):
                    xs_mb = xs_mb_all[i].cuda()
                    ys_mb = ys_mb_all[i].cuda()
                    g_xs_mb, f_g_xs_mb = base_network(xs_mb)

                    for j in range(k):
                        xt_mb       = xt_mb_all[j].cuda()
                        g_xt_mb, f_g_xt_mb = base_network(xt_mb)
                        pred_xt     = F.softmax(f_g_xt_mb, 1)
                        ys_oh       = F.one_hot(ys_mb, num_classes=class_num).float()
                        M_embed     = torch.cdist(g_xs_mb, g_xt_mb) ** 2
                        M_sce       = -torch.mm(ys_oh, torch.log(pred_xt + 1e-10).T)
                        M           = eta1 * M_embed + eta2 * M_sce
                        a_dist      = ot.unif(g_xs_mb.size(0))
                        b_dist      = ot.unif(g_xt_mb.size(0))
                        M_cpu       = M.detach().cpu().numpy()

                        # BoMb runs unguided: --use_bomb with --use_kpg is refused above,
                        # because Sec. IV's estimator is the uniform average of Eq. (8).
                        pi = solve_ot(a_dist, b_dist, M_cpu, ot_type, epsilon, tau, mass)

                        pi = torch.from_numpy(pi).float().cuda()
                        transfer_loss = torch.sum(pi * M)
                        list_transfer_loss.append(transfer_loss)

                # -- solve k×k OT between mini-batches -------------------
                big_C = torch.stack(list_transfer_loss).view(k, k)
                if be == 0:
                    plan = ot.emd([], [], big_C.detach().cpu().numpy())
                else:
                    plan = ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=be)

            # -- re-forward (with grad): compute gradients ---------------
            optimizer = lr_scheduler(optimizer, id_iter, **schedule_param)
            optimizer.zero_grad()

            for i in range(k):
                for j in range(k):
                    total_loss = 0
                    xs_mb = xs_mb_all[i].cuda()
                    ys_mb = ys_mb_all[i].cuda()
                    g_xs_mb, f_g_xs_mb = base_network(xs_mb)

                    classifier_loss = (
                        1.0 / (k ** 2) * nn.CrossEntropyLoss()(f_g_xs_mb, ys_mb)
                    )
                    total_loss += classifier_loss

                    if plan[i, j] == 0:
                        total_loss.backward()
                        continue

                    xt_mb       = xt_mb_all[j].cuda()
                    g_xt_mb, f_g_xt_mb = base_network(xt_mb)
                    pred_xt     = F.softmax(f_g_xt_mb, 1)
                    ys_oh       = F.one_hot(ys_mb, num_classes=class_num).float()
                    M_embed     = torch.cdist(g_xs_mb, g_xt_mb) ** 2
                    M_sce       = -torch.mm(ys_oh, torch.log(pred_xt + 1e-10).T)
                    M           = eta1 * M_embed + eta2 * M_sce
                    a_dist      = ot.unif(g_xs_mb.size(0))
                    b_dist      = ot.unif(g_xt_mb.size(0))
                    M_cpu       = M.detach().cpu().numpy()

                    # BoMb runs unguided (see note above).
                    pi = solve_ot(a_dist, b_dist, M_cpu, ot_type, epsilon, tau, mass)

                    pi            = torch.from_numpy(pi).float().cuda()
                    transfer_loss = torch.sum(pi * M)
                    transfer_loss = plan[i, j] * transfer_loss
                    total_loss   += transfer_loss
                    total_loss.backward()

            optimizer.step()

        # ================================================================
        # Standard averaging path
        # ================================================================
        else:
            optimizer = lr_scheduler(optimizer, id_iter, **schedule_param)
            optimizer.zero_grad()

            for i in range(k):
                total_loss = 0
                xs_mb = xs_mb_all[i].cuda()
                ys_mb = ys_mb_all[i].cuda()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    g_xs_mb, f_g_xs_mb = base_network(xs_mb)

                    classifier_loss = (
                        1.0 / k * nn.CrossEntropyLoss()(f_g_xs_mb, ys_mb)
                    )
                    total_loss += classifier_loss

                    xt_mb       = xt_mb_all[i].cuda()
                    g_xt_mb, f_g_xt_mb = base_network(xt_mb)
                    pred_xt     = F.softmax(f_g_xt_mb, 1)
                    ys_oh       = F.one_hot(ys_mb, num_classes=class_num).float()
                    M_embed     = torch.cdist(g_xs_mb, g_xt_mb) ** 2
                    M_sce       = -torch.mm(ys_oh, torch.log(pred_xt + 1e-10).T)
                    M           = eta1 * M_embed + eta2 * M_sce
                a_dist      = ot.unif(g_xs_mb.size(0))
                b_dist      = ot.unif(g_xt_mb.size(0))
                # .float(): the plan solvers expect fp32/64; under AMP, M is fp16.
                M_cpu       = M.detach().float().cpu().numpy()

                if kp_bank is not None:
                    # Sec. IV formulation: mask (Eq. 6) + guiding matrix (Eq. 9) +
                    # blended cost (Eq. 10), solved on the masked support.  Alg. 1,
                    # lines 4-14.  Identical guidance in all three regimes
                    # (Sec. IV-B: guidance is orthogonal to the marginal constraint):
                    #   unbalanced -> m-KUOT, derived scaling algorithm, Eq. (15)
                    #   balanced   -> m-KOT,  masked exact LP / Sinkhorn
                    #   partial    -> m-KPOT, masked partial OT with mass `mass`
                    #
                    # eps: ONLY the unbalanced solver requires eps > 0.  Balanced and
                    # partial must keep eps as configured (Table 1 uses 0 for them, i.e.
                    # the exact solvers) — falling back to kuot_eps there would silently
                    # switch m-KOT / m-KPOT to their entropic variants.
                    eps_eff = epsilon
                    if ot_type == "unbalanced" and eps_eff <= 0:
                        eps_eff = config["kuot_eps"]
                    pi = guided_plan(
                        M_cpu,
                        use_mask=not config["no_mask"],
                        feat_s=g_xs_mb.detach().cpu().numpy().astype(np.float64),
                        feat_t=g_xt_mb.detach().cpu().numpy().astype(np.float64),
                        k=kp_bank.k, alpha=alpha, rho=config["rho"],
                        metric=config["kp_metric"], ot_type=ot_type,
                        eps=eps_eff,
                        tau=tau, mass=mass, n_iter=config["kuot_iters"],
                        return_numpy=True,
                    )
                else:
                    pi = solve_ot(a_dist, b_dist, M_cpu, ot_type, epsilon, tau, mass)

                pi            = torch.from_numpy(pi).float().cuda()
                transfer_loss = torch.sum(pi * M.float())
                transfer_loss = 1.0 / k * transfer_loss
                total_loss   += transfer_loss
                amp_scaler.scale(total_loss).backward()

            amp_scaler.step(optimizer)
            amp_scaler.update()

    checkpoint = {"base_network": temp_model.state_dict()}
    torch.save(checkpoint, osp.join(config["output_path"], "final_model.pth"))
    return best_acc


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser = argparse.ArgumentParser(
        description="Mini-batch Keypoint-Guided Optimal Transport for Deep Domain Adaptation"
    )

    # ---- general -------------------------------------------------------
    parser.add_argument("--gpu_id",      type=str,    nargs="?", default="0")
    parser.add_argument(
        "--net", type=str, default="ResNet50",
        choices=[
            "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
            "VGG11", "VGG13", "VGG16", "VGG19",
            "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet",
        ],
    )
    parser.add_argument(
        "--dset", type=str, default="office",
        choices=["office", "image-clef", "visda", "office-home"],
    )
    parser.add_argument("--s_dset_path",     type=str, default="./data/office/amazon_31_list.txt")
    parser.add_argument("--t_dset_path",     type=str, default="./data/office/webcam_10_list.txt")
    parser.add_argument("--stratify_source", action="store_true")
    parser.add_argument("--test_interval",   type=int, default=500)
    parser.add_argument("--output_dir",      type=str, default="san")
    parser.add_argument("--restore_dir",     type=str, default=None)
    parser.add_argument("--lr",              type=float, default=0.001)
    parser.add_argument("--batch_size",      type=int, default=36)
    parser.add_argument("--cos_dist",        type=str2bool, default=False)
    parser.add_argument("--stop_step",       type=int, default=0)
    parser.add_argument("--final_log",       type=str, default=None)
    parser.add_argument("--seed",            type=int, default=12345)
    parser.add_argument("--num_worker",      type=int, default=4)
    parser.add_argument("--test_10crop",     type=str2bool, default=True)

    # ---- Mini-batch OT (BoMb) parameters -------------------------------
    parser.add_argument(
        "--ot_type", type=str, default="balanced",
        choices=["balanced", "unbalanced", "partial"],
        help="Type of optimal transport",
    )
    parser.add_argument("--eta1",    type=float, default=0.1,  help="weight of embedding cost")
    parser.add_argument("--eta2",    type=float, default=0.1,  help="weight of SCE cost")
    parser.add_argument("--epsilon", type=float, default=0.0,  help="OT entropic regularization (0 = exact)")
    parser.add_argument("--tau",     type=float, default=1.0,  help="marginal penalisation (unbalanced OT)")
    parser.add_argument(
        "--mass", type=float, default=0.65,
        help="transported mass ratio s (partial OT).  0.65 on Office-Home and VisDA-2017, "
             "and held fixed there -- only the guidance weight --alpha is varied.  (Digits "
             "m-KPOT is the one exception: it uses a per-task s of 0.85 / 0.90 / 0.80, set "
             "in DeepDA/digits/sh/train_KPG_mPOT.sh.)")
    parser.add_argument("--use_bomb",action="store_true",      help="use BoMb hierarchical scheme")
    parser.add_argument("--be",      type=float, default=0.0,  help="inter-batch OT regularization")
    parser.add_argument("--k",       type=int,   default=1,    help="number of mini-batches per update")

    # ---- KPG-RL parameters ---------------------------------------------
    parser.add_argument(
        "--use_kpg", action="store_true",
        help="enable KPG-RL keypoint-guided OT",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="KPG-RL-KP combination coefficient: "
             "cost = alpha * C_norm + (1 - alpha) * G_norm.  "
             "(1 = pure standard cost, 0 = pure KPG guiding cost)",
    )

    # ---- Sec. IV formulation ---------------------------
    parser.add_argument(
        "--kp_strategy", type=str, default="random",
        choices=["centroid", "random", "farthest"],
        help="Keypoint-selection strategy (Sec. V-A-3).  "
             "'centroid' = target medoid of the true class (ORACLE, uses "
             "GT target labels), 'random' = random pseudo-class sample (practical), "
             "'farthest' = pseudo-class sample farthest from its centroid (adversarial).",
    )
    parser.add_argument(
        "--kp_per_class", type=int, default=1,
        help="Keypoint pairs per class.  k = kp_per_class x #classes; must satisfy k < m.",
    )
    parser.add_argument(
        "--kp_n_classes", type=int, default=None,
        help="Restrict keypoints to the first N classes (PDA-style shared-class subset). "
             "Also the practical way to keep k < m when #classes >= batch size.",
    )
    parser.add_argument(
        "--rho", type=float, default=0.1,
        help="Dimensionless softmax temperature of the relation profiles, Eq. (8): the "
             "softmax scale is rho * max(c).",
    )
    parser.add_argument(
        "--kp_metric", type=str, default="euclidean",
        choices=["euclidean", "sqeuclidean"],
        help="Ground metric for the relation profiles of Eq. (8).",
    )
    parser.add_argument(
        "--kp_probe", type=int, default=4096,
        help="Number of samples per domain encoded once to choose the fixed keypoints.",
    )
    parser.add_argument(
        "--no_mask", action="store_true",
        help="A1 mask-off arm: keep the keypoints and the guiding matrix but replace the "
             "mask of Eq. (6) by an all-ones matrix, so no pair is pinned.  Isolates the "
             "mask's contribution from the guiding cost's.",
    )
    parser.add_argument(
        "--kp_replace", action="store_true",
        help="Sample the m-k non-keypoint slots with replacement (Sec. IV-A permits either).",
    )
    parser.add_argument(
        "--kuot_eps", type=float, default=0.01,
        help="Entropic regularisation of Eq. (12) used by the derived scaling solver when "
             "--epsilon is 0.  The paper uses 0.01.",
    )
    parser.add_argument(
        "--kuot_iters", type=int, default=1000,
        help="Scaling iterations N in Alg. 1.",
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # ---- reproducibility -----------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # ---- build config --------------------------------------------------
    config = {}
    config["args"]           = args
    config["gpu"]            = args.gpu_id
    config["num_iterations"] = args.stop_step + 1
    config["test_interval"]  = args.test_interval
    config["ot_type"]        = args.ot_type
    config["eta1"]           = args.eta1
    config["eta2"]           = args.eta2
    config["epsilon"]        = args.epsilon
    config["tau"]            = args.tau
    config["mass"]           = args.mass
    config["use_bomb"]       = args.use_bomb
    config["be"]             = args.be
    config["k"]              = args.k
    config["use_kpg"]        = args.use_kpg
    config["alpha"]          = args.alpha
    # Sec. IV formulation
    config["kp_strategy"]    = args.kp_strategy
    config["kp_per_class"]   = args.kp_per_class
    config["kp_n_classes"]   = args.kp_n_classes
    config["rho"]            = args.rho
    config["kp_metric"]      = args.kp_metric
    config["no_mask"]        = args.no_mask
    config["kp_probe"]       = args.kp_probe
    config["kp_replace"]     = args.kp_replace
    config["kuot_eps"]       = args.kuot_eps
    config["kuot_iters"]     = args.kuot_iters
    config["output_for_test"] = True
    config["output_path"]    = "snapshot/" + args.output_dir
    config["restore_path"]   = "snapshot/" + args.restore_dir if args.restore_dir else None

    if os.path.exists(config["output_path"]):
        print("checkpoint dir exists, which will be removed")
        import shutil
        shutil.rmtree(config["output_path"], ignore_errors=True)
    if not os.path.isdir("snapshot/"):
        os.mkdir("snapshot/")
    os.mkdir(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")

    config["prep"] = {
        "test_10crop": args.test_10crop,
        "params":      {"resize_size": 256, "crop_size": 224},
    }

    if "ResNet" in args.net:
        config["network"] = {
            "name":   network.ResNetFc,
            "params": {
                "resnet_name":    args.net,
                "use_bottleneck": True,
                "bottleneck_dim": 512,
                "new_cls":        True,
                "cos_dist":       args.cos_dist,
            },
        }
    elif "VGG" in args.net:
        config["network"] = {
            "name":   network.VGGFc,
            "params": {
                "vgg_name":       args.net,
                "use_bottleneck": True,
                "bottleneck_dim": 256,
                "new_cls":        True,
            },
        }

    config["optimizer"] = {
        "type":         optim.SGD,
        "optim_params": {
            "lr": args.lr, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": True
        },
        "lr_type":  "inv",
        "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75},
    }

    config["dataset"] = args.dset
    test_bs = 4
    if config["dataset"] == "office":
        if (
            ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path)
            or ("webcam" in args.s_dset_path and "dslr"   in args.t_dset_path)
            or ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path)
            or ("dslr"   in args.s_dset_path and "amazon" in args.t_dset_path)
        ):
            config["optimizer"]["lr_param"]["lr"] = 0.001
        elif ("amazon" in args.s_dset_path and "dslr"   in args.t_dset_path) or (
             "dslr"   in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003
            args.stop_step = 20000
        else:
            config["optimizer"]["lr_param"]["lr"] = 0.001
        config["network"]["params"]["class_num"] = 31
        args.stop_step = 20000
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"]    = 0.001
        config["network"]["params"]["class_num"] = 65
        test_bs = 10
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"]    = 0.001
        config["network"]["params"]["class_num"] = 12
        test_bs = 61
    else:
        raise ValueError("Dataset has not been implemented.")

    config["data"] = {
        "source": {"list_path": args.s_dset_path, "batch_size": args.batch_size},
        "target": {"list_path": args.t_dset_path, "batch_size": args.batch_size},
        "test":   {"list_path": args.t_dset_path, "batch_size": test_bs},
    }

    if args.lr != 0.001:
        config["optimizer"]["lr_param"]["lr"]    = args.lr
        config["optimizer"]["lr_param"]["gamma"] = 0.001
    config["out_file"].write(str(config) + "\n")
    config["out_file"].flush()
    config["stop_step"] = args.stop_step if args.stop_step != 0 else 10000

    if args.final_log is None:
        config["final_log"] = open("log.txt", "a")
    else:
        config["final_log"] = open(args.final_log, "a")

    train(config)
