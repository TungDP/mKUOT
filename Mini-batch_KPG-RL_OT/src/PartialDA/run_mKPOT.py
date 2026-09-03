import argparse
import os
import random
import data_list
import lr_schedule
import network
import numpy as np
import ot
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import BalancedBatchSampler

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

def guided_plan(cost, feat_s, feat_t, k, alpha=0.5, rho=0.1, metric="euclidean",
                ot_type="unbalanced", eps=0.01, tau=1.0, mass=1.0, n_iter=200,
                return_numpy=False):
    """Solve KUOT on one mini-batch and return the plan.

    `cost` is the task cost C (torch tensor or numpy array) with the keypoints in the first
    k rows/columns.  `feat_s` / `feat_t` are the matching batch features.

    Steps (Alg. 1): build the mask -> relation profiles + guiding matrix -> blend -> masked
    solve.  With k == 0 this degrades to the plain unguided masked-free problem.
    """
    is_torch = torch.is_tensor(cost)
    C_np = cost.detach().cpu().numpy().astype(np.float64) if is_torch else np.asarray(cost, dtype=np.float64)

    m = C_np.shape[0]
    M = build_mask(m, k)
    G = guiding_matrix(feat_s, feat_t, k, rho=rho, metric=metric)
    C_tilde = blend_cost(C_np, G, alpha)
    T = solve_kuot(C_tilde, M, ot_type=ot_type, eps=eps, tau=tau, mass=mass,
                        n_iter=n_iter)

    if return_numpy or not is_torch:
        return T
    return torch.from_numpy(T).float().to(cost.device)


def image_train(resize_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def image_classification(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader["test"])):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = (
        torch.sum(torch.squeeze(predict).float() == all_label).item()
        / float(all_label.size()[0])
    )
    return accuracy


# -----------------------------------------------------------------------
# Unguided OT solvers -- the m-OT / m-UOT / m-POT baselines.
# The guided path is the Sec. IV block above (solve_kuot / solve_kuot_paper).
# -----------------------------------------------------------------------

def solve_ot(a, b, M_norm, ot_type, epsilon, tau, adap_mass):
    """Standard OT dispatcher (ot / uot / pot).

    Note on entropic POT stability: `ot.partial.entropic_partial_wasserstein`
    can return NaN when `adap_mass` is very small (~0.01 during the warm-up
    of the linear ramp) combined with small `epsilon` — the multiplicative
    Sinkhorn updates `q1 * Kprev / K1` produce 0/0.  When that happens we
    fall back to the exact LP solver `ot.partial.partial_wasserstein`, which
    is unconditionally stable.
    """
    if ot_type == "ot":
        if epsilon == 0:
            return ot.emd(a, b, M_norm)
        else:
            return ot.sinkhorn(a, b, M_norm, reg=epsilon)
    elif ot_type == "uot":
        return ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M_norm, epsilon, tau)
    elif ot_type == "pot":
        if epsilon == 0:
            return ot.partial.partial_wasserstein(a, b, M_norm, adap_mass)
        pi = ot.partial.entropic_partial_wasserstein(
            a, b, M_norm, m=adap_mass, reg=epsilon
        )
        if np.any(np.isnan(pi)):
            # Sinkhorn diverged — fall back to exact LP
            return ot.partial.partial_wasserstein(a, b, M_norm, adap_mass)
        return pi


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train(args):
    eta1     = args.eta1
    eta2     = args.eta2
    eta3     = args.eta3
    tau      = args.tau
    epsilon  = args.epsilon
    mass     = args.mass
    k        = args.k
    ot_type  = args.ot_type
    use_kpg  = args.use_kpg
    alpha    = args.alpha
    n_shared_classes = args.n_shared_classes
    # Seeded RNG so the "random" target strategy is reproducible across runs
    kpg_rng = np.random.default_rng(args.seed)

    log_str = (
        "-" * 50 + "\n"
        " eta1 = {:.3f}, eta2 = {:.3f}, eta3 = {:.3f}\n"
        " ot_type = {}, epsilon = {:.4f}, tau = {:.4f}, mass = {:.3f}\n"
        " use_kpg = {}, alpha = {:.3f}, rho = {:.3f}\n"
        " n_shared_classes = {} / class_num = {}\n"
        " kp_strategy = {}  (centroid=oracle, random=practical, farthest=adversarial)\n"
    ).format(eta1, eta2, eta3, ot_type, epsilon, tau, mass,
             use_kpg, alpha, args.rho,
             n_shared_classes, args.class_num, args.kp_strategy)
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    train_bs, test_bs = args.batch_size, args.batch_size * 2

    dsets = {}
    dsets["source"] = data_list.ImageList(
        open(args.s_dset_path).readlines(), transform=image_train()
    )
    dsets["target"] = data_list.ImageList(
        open(args.t_dset_path).readlines(), transform=image_train()
    )
    dsets["test"] = data_list.ImageList(
        open(args.t_dset_path).readlines(), transform=image_test()
    )

    dset_loaders = {}

    # Balanced sampler for source: ensures each class appears every batch
    source_labels = torch.zeros(len(dsets["source"]))
    for i, line in enumerate(open(args.s_dset_path).readlines()):
        source_labels[i] = int(line.split()[1])
    train_batch_sampler = BalancedBatchSampler(source_labels, batch_size=train_bs)
    dset_loaders["source"] = DataLoader(
        dsets["source"], batch_sampler=train_batch_sampler, num_workers=args.worker
    )

    dset_loaders["target"] = DataLoader(
        dsets["target"], batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=True,
    )
    dset_loaders["test"] = DataLoader(
        dsets["test"], batch_size=test_bs, shuffle=False, num_workers=args.worker
    )

    if "ResNet" in args.net:
        params = {
            "resnet_name": args.net,
            "use_bottleneck": True,
            "bottleneck_dim": 256,
            "new_cls": True,
            "class_num": args.class_num,
        }
        base_network = network.ResNetFc(**params)
    elif "VGG" in args.net:
        params = {
            "vgg_name": args.net,
            "use_bottleneck": True,
            "bottleneck_dim": 256,
            "new_cls": True,
            "class_num": args.class_num,
        }
        base_network = network.VGGFc(**params)

    base_network = base_network.cuda()
    parameter_list = base_network.get_parameters()
    base_network = torch.nn.DataParallel(base_network).cuda()

    optimizer_config = {
        "type": torch.optim.SGD,
        "optim_params": {
            "lr": args.lr, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": True
        },
        "lr_type": "inv",
        "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75},
    }
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    iter_source = iter(dset_loaders["source"])
    iter_target = iter(dset_loaders["target"])
    best_acc  = 0.0
    best_iter = 0

    # ---- Sec. IV formulation: select the FIXED keypoint pairs once ------
    kp_bank = None
    if use_kpg:
        kp_bank = KeypointBank(
            strategy=args.kp_strategy, kp_per_class=args.kp_per_class,
            rho=args.rho, metric=args.kp_metric, seed=args.seed,
        )
        probe_rng = np.random.default_rng(args.seed)
        pool_s = np.sort(probe_rng.choice(
            len(dsets["source"]), size=min(args.kp_probe, len(dsets["source"])), replace=False))
        pool_t = np.sort(probe_rng.choice(
            len(dsets["target"]), size=min(args.kp_probe, len(dsets["target"])), replace=False))

        base_network.eval()
        feat_s = extract_features(base_network, dsets["source"], pool_s)
        feat_t, logit_t = extract_features(
            base_network, dsets["target"], pool_t, want_logits=True)
        base_network.train()

        lab_s = np.array([int(dsets["source"][int(i)][1]) for i in pool_s])
        if args.kp_strategy == "centroid":
            lab_t = np.array([int(dsets["target"][int(j)][1]) for j in pool_t])
        else:
            lab_t = logit_t.argmax(axis=1)

        # PDA: keypoints only for the shared classes (Sec. V-C).
        n_kp_cls = n_shared_classes if n_shared_classes is not None else args.class_num
        k_eff = kp_bank.select(feat_s, lab_s, feat_t, lab_t, range(n_kp_cls),
                              pool_s=pool_s, pool_t=pool_t)
        print("[mkuot] " + kp_bank.describe())
        args.out_file.write("[mkuot] " + kp_bank.describe() + "\n")
        args.out_file.flush()
        if k_eff == 0:
            raise RuntimeError("keypoint selection produced 0 pairs; raise --kp_probe")
        check_batch_feasibility(train_bs, k_eff, strict=True)

    for i in tqdm(range(args.max_iterations + 1)):

        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iterations):
            base_network.train(False)
            temp_acc = image_classification(dset_loaders, base_network)
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str)
            if best_acc < temp_acc:
                best_acc  = temp_acc
                best_iter = i

        if i % args.test_interval == 0:
            log_str = "\n{}, iter: {:05d}, source/target: {:02d}/{:02d}\n".format(
                args.name, i, train_bs, train_bs
            )
            args.out_file.write(log_str)
            args.out_file.flush()
            print(log_str)

        base_network.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        # Transported mass is FIXED at `mass` (s = 0.65) in every setting -- only the
        # guidance weight alpha is varied.  The former linear ramp from 0.01 is gone:
        # it made s time-varying (so not comparable across runs) and it was the source
        # of the entropic-POT NaNs that solve_ot() had to guard against, since those
        # appear only at very small transported mass.
        adap_mass = mass

        for _ in range(k):
            try:
                xs, ys = next(iter_source)
                xt, yt = next(iter_target)
            except StopIteration:
                iter_source = iter(dset_loaders["source"])
                iter_target = iter(dset_loaders["target"])
                xs, ys = next(iter_source)
                xt, yt = next(iter_target)

            xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()
            if kp_bank is not None:
                # Eq. (5): the k fixed keypoint pairs occupy the first k slots.
                xs, ys, xt, yt = inject_keypoints(
                    kp_bank, xs, ys, xt, yt, dsets["source"], dsets["target"])
            # Target labels stay on CPU — only used (if at all) for the
            # `centroid` oracle keypoint strategy.
            yt_np = yt.numpy() if args.kp_strategy == "centroid" else None

            g_xs, f_g_xs = base_network(xs)
            g_xt, f_g_xt = base_network(xt)

            pred_xt = F.softmax(f_g_xt, 1)

            classifier_loss = torch.nn.CrossEntropyLoss()(f_g_xs, ys) / k

            ys_oh   = F.one_hot(ys, num_classes=args.class_num).float()
            M_embed = torch.cdist(g_xs, g_xt) ** 2
            M_sce   = -torch.mm(ys_oh, torch.log(pred_xt + 1e-10).T)
            M       = eta1 * M_embed + eta2 * M_sce

            a      = ot.unif(g_xs.size(0))
            b      = ot.unif(g_xt.size(0))
            M_cpu  = M.detach().cpu().numpy()

            if use_kpg and kp_bank is not None:
                # Sec. IV: mask (Eq. 6) + guiding matrix (Eq. 9) + blend (Eq. 10), solved
                # by the derived scaling algorithm (Eq. 15) in the unbalanced regime.
                # For PDA the keypoints cover only the shared classes, so no keypoint is
                # ever assigned to a source-private category (Sec. V-C).
                # Only the unbalanced solver needs eps > 0.  'ot' (m-KOT) and 'pot'
                # (m-KPOT) keep eps as configured (0 = exact solvers), so the balanced
                # and partial regimes are not silently switched to entropic variants.
                # `adap_mass` is the ramped transported mass for the partial regime.
                eps_eff = epsilon
                if ot_type == "uot" and eps_eff <= 0:
                    eps_eff = args.kuot_eps
                pi = guided_plan(
                    M_cpu,
                    feat_s=g_xs.detach().cpu().numpy().astype(np.float64),
                    feat_t=g_xt.detach().cpu().numpy().astype(np.float64),
                    k=kp_bank.k, alpha=alpha, rho=args.rho,
                    metric=args.kp_metric, ot_type=ot_type,
                    eps=eps_eff,
                    tau=tau, mass=adap_mass, n_iter=args.kuot_iters,
                    return_numpy=True,
                )
            else:
                M_norm = M_cpu / (M_cpu.max() + 1e-8)
                pi = solve_ot(a, b, M_norm, ot_type, epsilon, tau, adap_mass)

            pi = torch.from_numpy(pi).float().cuda()
            transfer_loss = eta3 * torch.sum(pi * M) / k

            if i % 100 == 0:
                # Number of active source centroids = #shared classes present
                # in this mini-batch.
                if use_kpg:
                    n_kp = len(np.unique(ys.cpu().numpy()[
                        ys.cpu().numpy() < n_shared_classes
                    ]))
                else:
                    n_kp = "N/A"
                log_str = (
                    "sum(pi)={:.4f}, transfer={:.4f}, "
                    "adap_mass={:.3f}, kps={}\n"
                ).format(
                    torch.sum(pi).item(), transfer_loss.item(),
                    adap_mass, n_kp,
                )
                args.out_file.write(log_str)
                args.out_file.flush()
                print(log_str)

            total_loss = classifier_loss + transfer_loss
            total_loss.backward()

        optimizer.step()

    log_str = "Acc: {:.2f}\n".format(np.round(best_acc * 100, 2))
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    result_tag = "kpg_" + ot_type if use_kpg else ot_type
    result_tag += "_alpha{:.1f}".format(alpha) if use_kpg else ""
    result_tag += f"_{args.kp_strategy}"
    with open(f"results/result_m{result_tag}.txt", "a") as f:
        f.write(
            "method {}, iter: {:05d}, precision: {:.5f}\n".format(
                args.name + "_" + args.output, best_iter, best_acc
            )
        )

    return best_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Mini-batch Keypoint-Guided OT for Partial Domain Adaptation"
    )
    parser.add_argument("--gpu_id",   type=str, nargs="?", default="0")
    parser.add_argument("--s",        type=int, default=0,  help="source domain index")
    parser.add_argument("--t",        type=int, default=1,  help="target domain index")
    parser.add_argument("--output",   type=str, default="run")
    parser.add_argument("--seed",     type=int, default=2020)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--batch_size",     type=int, default=65)
    parser.add_argument("--worker",         type=int, default=4)
    parser.add_argument("--net",      type=str, default="ResNet50",
                        choices=["ResNet50", "VGG16"])
    parser.add_argument("--dset",     type=str, default="office_home",
                        choices=["office_home"])
    parser.add_argument("--test_interval", type=int, default=500)
    parser.add_argument("--lr",       type=float, default=0.001)

    # OT parameters
    parser.add_argument("--ot_type",  type=str, default="pot",
                        choices=["ot", "uot", "pot"])
    parser.add_argument("--eta1",     type=float, default=0.003)
    parser.add_argument("--eta2",     type=float, default=0.75)
    parser.add_argument("--eta3",     type=float, default=10.0)
    parser.add_argument("--epsilon",  type=float, default=0.0)
    parser.add_argument("--tau",      type=float, default=0.06)
    parser.add_argument("--mass",     type=float, default=0.65,
                        help="transported mass s (partial OT), fixed at 0.65 in every "
                             "setting; only --alpha is varied")
    parser.add_argument("--k",        type=int,   default=1)

    # KPG-RL parameters
    parser.add_argument("--use_kpg",  action="store_true",
                        help="enable KPG-RL keypoint-guided OT")
    parser.add_argument("--alpha",    type=float, default=0.9,
                        help="cost = alpha*C_norm + (1-alpha)*G")
    # ---- Sec. IV formulation ---------------------------
    parser.add_argument("--kp_strategy", type=str, default="random",
                        choices=["centroid", "random", "farthest"],
                        help="keypoint-selection strategy (Sec. V-A-3): centroid=oracle, "
                             "random=practical, farthest=adversarial")
    parser.add_argument("--kp_per_class", type=int, default=1,
                        help="keypoint pairs per shared class; k = kp_per_class x classes, k < m")
    parser.add_argument("--rho", type=float, default=0.1,
                        help="dimensionless relation-profile temperature, Eq. (8) "
                             "(softmax scale rho*max(c))")
    parser.add_argument("--kp_metric", type=str, default="euclidean",
                        choices=["euclidean", "sqeuclidean"],
                        help="ground metric for the relation profiles of Eq. (8)")
    parser.add_argument("--kp_probe", type=int, default=4096,
                        help="samples per domain encoded once to choose the fixed keypoints")
    parser.add_argument("--kuot_eps", type=float, default=0.01,
                        help="entropic regularisation of Eq. (12) when --epsilon is 0")
    parser.add_argument("--kuot_iters", type=int, default=1000,
                        help="scaling iterations N in Alg. 1")

    parser.add_argument("--n_shared_classes", type=int, default=None,
                        help="restrict KPG keypoint candidate classes to "
                             "0..n_shared_classes-1.  Required for PDA "
                             "(set to 25 for Office-Home PDA).  None means "
                             "consider all class_num classes (closed-set DA).")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dset == "office_home":
        names = ["Art", "Clipart", "Product", "RealWorld"]
        args.class_num   = 65
        args.max_iterations = 5000
        args.test_interval  = 500

    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    args.s_dset_path = os.path.join(data_folder, args.dset, names[args.s] + "_list.txt")
    args.t_dset_path = os.path.join(data_folder, args.dset, names[args.t] + "_25_list.txt")

    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    args.output_dir = os.path.join("snapshot", args.name, args.output)
    os.makedirs(args.output_dir, exist_ok=True)
    args.out_file = open(os.path.join(args.output_dir, "log.txt"), "w")
    args.out_file.write(str(args) + "\n")
    args.out_file.flush()

    train(args)
