"""
Mini-batch Keypoint-Guided Optimal Transport for Deep Domain Adaptation (digits).

Implements Section IV of the paper: the k FIXED keypoint pairs occupy the first k
slots of every mini-batch (Eq. 5), a binary mask pins each pair to its partner
(Eq. 6-7), relation profiles to the keypoints give the guiding matrix (Eq. 8-9),
and the blended cost (Eq. 10) is solved on the masked support -- by the derived
matrix-scaling algorithm (Eq. 12-15) in the unbalanced regime.

Keypoint pairs are chosen once before training by `--kp_strategy`
(centroid = oracle | random = practical | farthest = adversarial).

When --use_kpg is not set, behaviour is identical to the baseline methods.py.
"""

import os

import numpy as np
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from utils import model_eval, save_acc

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


# -----------------------------------------------------------------------
# KPG-RL helper functions
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# OT dispatch
# -----------------------------------------------------------------------

def _solve_ot(a, b, C_np, method, epsilon, tau, mass):
    """Standard OT solver (no KPG)."""
    if method == "jumbot":
        return ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, C_np, epsilon, tau)
    elif method == "jdot":
        if epsilon == 0:
            return ot.emd(a, b, C_np)
        else:
            return ot.sinkhorn(a, b, C_np, reg=epsilon)
    elif method == "jpmbot":
        if epsilon == 0:
            return ot.partial.partial_wasserstein(a, b, C_np, mass)
        else:
            _M = C_np / (C_np.max() + 1e-10)
            return ot.partial.entropic_partial_wasserstein(a, b, _M, m=mass, reg=epsilon)


# -----------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------

class DigitsDA:
    def __init__(
        self, model_g, model_f, n_class, logger, out_dir,
        eta1=0.1, eta2=0.1, epsilon=0.1, batch_epsilon=0.0,
        mass=0.5, tau=1.0, test_interval=10,
        use_kpg=False, alpha=0.5, kpg_seed=1980,
        kp_bank=None, rho=0.1, kp_metric="euclidean",
        kuot_eps=0.01, kuot_iters=1000,
        kp_strategy="random", kp_per_class=1, kp_probe=4096,
    ):
        self.model_g = model_g
        self.model_f = model_f
        self.n_class = n_class
        self.logger = logger
        self.out_dir = out_dir
        self.out_file = os.path.join(self.out_dir, "acc.csv")
        if os.path.exists(self.out_file):
            os.remove(self.out_file)
        self.eta1 = eta1
        self.eta2 = eta2
        self.epsilon = epsilon
        self.batch_epsilon = batch_epsilon
        self.mass = mass
        self.tau = tau
        self.test_interval = test_interval
        self.use_kpg = use_kpg
        self.alpha = alpha
        # RNG for the target-keypoint strategy (reproducible).
        self.kpg_rng   = np.random.default_rng(kpg_seed)
        # Sec. IV formulation (mask + fixed keypoints + derived scaling solver).
        self.kp_bank     = kp_bank
        self.rho         = rho
        self.kp_metric   = kp_metric
        self.kuot_eps    = kuot_eps
        self.kuot_iters  = kuot_iters
        self.kp_strategy  = kp_strategy
        self.kp_per_class = kp_per_class
        self.kp_probe     = kp_probe
        self.kp_seed      = kpg_seed
        self.logger.info(
            "eta1={}, eta2={}, epsilon={}, use_kpg={}, alpha={}, "
            "rho={}, kp_strategy={}, kp_per_class={}".format(
                eta1, eta2, epsilon, use_kpg, alpha, rho,
                kp_strategy, kp_per_class,
            )
        )

    # ----- inner OT helper (shared by all training paths) ---------------

    def _inner_ot(self, g_xs, g_xt, ys, pred_xt, total_cost, method, yt=None):
        """Solve the inner OT and return the plan as a cuda tensor.

        `yt` (GT target labels) is required only when `kp_strategy == "centroid"`.
        """
        a = ot.unif(g_xs.size(0))
        b = ot.unif(g_xt.size(0))
        C_np = total_cost.detach().cpu().numpy()

        if self.use_kpg and self.kp_bank is not None:
            # Sec. IV: mask (Eq. 6) + guiding matrix (Eq. 9) + blend (Eq. 10), solved on
            # the masked support.  Keypoints occupy the first k slots (Eq. 5).  The same
            # guidance applies in all three regimes (Sec. IV-B):
            #   jumbot -> m-KUOT (unbalanced, derived scaling algorithm Eq. 15)
            #   jdot   -> m-KOT  (balanced, masked exact LP / Sinkhorn)
            #   jpmbot -> m-KPOT (partial, masked partial OT with mass `self.mass`)
            #
            # Only the unbalanced solver needs eps > 0; jdot / jpmbot keep eps as given
            # (0 = exact) so m-KOT / m-KPOT are not silently made entropic.
            eps_eff = self.epsilon
            if method == "jumbot" and eps_eff <= 0:
                eps_eff = self.kuot_eps
            pi = guided_plan(
                C_np,
                feat_s=g_xs.detach().cpu().numpy().astype(np.float64),
                feat_t=g_xt.detach().cpu().numpy().astype(np.float64),
                k=self.kp_bank.k, alpha=self.alpha, rho=self.rho,
                metric=self.kp_metric, ot_type=method,
                eps=eps_eff,
                tau=self.tau, mass=self.mass, n_iter=self.kuot_iters,
                return_numpy=True,
            )
            return torch.from_numpy(pi).float().cuda()

        pi = _solve_ot(a, b, C_np, method, self.epsilon, self.tau, self.mass)

        return torch.from_numpy(pi).float().cuda()

    def _compute_cost(self, g_xs, g_xt, ys, pred_xt):
        """Compute ground cost: eta1 * embed + eta2 * label."""
        embed_cost = torch.cdist(g_xs, g_xt) ** 2
        ys_oh = F.one_hot(ys, num_classes=self.n_class).float()
        t_cost = -torch.mm(ys_oh, torch.log(pred_xt + 1e-10).T)
        return self.eta1 * embed_cost + self.eta2 * t_cost

    # ----- fit: standard mOT (averaging) --------------------------------

    def build_keypoint_bank(self, source_loader, target_loader, batch_size):
        """Select the FIXED keypoint pairs of Eq. (4) once, before training (Sec. IV-A).

        Source labels are ground truth; target labels are the current classifier's
        pseudo-labels, except for the `centroid` oracle strategy which consumes ground-truth
        target labels.  Only the dataset indices are stored, so the pairs stay fixed while
        their embeddings move with the encoder.
        """
        bank = KeypointBank(
            strategy=self.kp_strategy, kp_per_class=self.kp_per_class,
            rho=self.rho, metric=self.kp_metric, seed=self.kp_seed,
        )
        dset_s, dset_t = source_loader.dataset, target_loader.dataset
        rng = np.random.default_rng(self.kp_seed)
        pool_s = np.sort(rng.choice(len(dset_s), size=min(self.kp_probe, len(dset_s)),
                                    replace=False))
        pool_t = np.sort(rng.choice(len(dset_t), size=min(self.kp_probe, len(dset_t)),
                                    replace=False))

        def _fwd(x):
            g = self.model_g(x)
            return g, self.model_f(g)

        self.model_g.eval(); self.model_f.eval()
        feat_s = extract_features(lambda x: _fwd(x)[0], dset_s, pool_s)
        feat_t, logit_t = extract_features(_fwd, dset_t, pool_t, want_logits=True)
        self.model_g.train(); self.model_f.train()

        lab_s = np.array([int(dset_s[int(i)][1]) for i in pool_s])
        if self.kp_strategy == "centroid":
            lab_t = np.array([int(dset_t[int(j)][1]) for j in pool_t])   # oracle
        else:
            lab_t = logit_t.argmax(axis=1)

        k_eff = bank.select(feat_s, lab_s, feat_t, lab_t, range(self.n_class),
                            pool_s=pool_s, pool_t=pool_t)
        self.logger.info("[mkuot] " + bank.describe())
        if k_eff == 0:
            raise RuntimeError("keypoint selection produced 0 pairs; raise --kp_probe")
        # The per-update mini-batch is batch_size // k_mb points; guard k < m on it.
        check_batch_feasibility(batch_size, k_eff, strict=True)
        self.kp_bank = bank
        return bank

    def fit(self, source_loader, target_loader, test_loader,
            n_epochs, criterion=nn.CrossEntropyLoss(), lr=2e-4,
            k=1, batch_size=25, method="jumbot"):
        criterion = nn.CrossEntropyLoss()
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)
        best_acc = 0

        # Sec. IV formulation: pick the fixed keypoint pairs once, then place them in the
        # first k slots of every sub-batch (Eq. (5)).
        if self.use_kpg and self.kp_bank is None:
            self.build_keypoint_bank(source_loader, target_loader, batch_size // k)

        for id_epoch in range(n_epochs):
            print(f"Epoch: {id_epoch}")
            self.model_g.train()
            self.model_f.train()
            target_loader_iter = iter(target_loader)

            for _, data in tqdm(enumerate(source_loader)):
                xs_mb_all, ys_all = data
                try:
                    xt_mb_all, yt_all = next(target_loader_iter)
                except StopIteration:
                    xt_mb_all = None
                if xt_mb_all is None or len(xt_mb_all) != batch_size:
                    target_loader_iter = iter(target_loader)
                    xt_mb_all, yt_all = next(target_loader_iter)

                inds_xs = np.split(np.arange(xs_mb_all.shape[0]), k)

                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                for i in range(k):
                    total_loss = 0
                    xs_mb = xs_mb_all[inds_xs[i]].cuda()
                    ys = ys_all[inds_xs[i]].cuda()
                    xt_mb = xt_mb_all[inds_xs[i]].cuda()
                    yt_mb = yt_all[inds_xs[i]]

                    if self.kp_bank is not None:
                        # Eq. (5): the k fixed keypoint pairs occupy the first k slots of
                        # this sub-batch, aligned across the two domains.
                        xs_mb, ys, xt_mb, yt_mb = inject_keypoints(
                            self.kp_bank, xs_mb, ys, xt_mb, yt_mb,
                            source_loader.dataset, target_loader.dataset,
                        )

                    g_xs_mb = self.model_g(xs_mb)
                    f_g_xs_mb = self.model_f(g_xs_mb)

                    s_loss = 1.0 / k * criterion(f_g_xs_mb, ys)
                    total_loss += s_loss

                    g_xt_mb = self.model_g(xt_mb)
                    f_g_xt_mb = self.model_f(g_xt_mb)
                    pred_xt = F.softmax(f_g_xt_mb, 1)

                    total_cost = self._compute_cost(g_xs_mb, g_xt_mb, ys, pred_xt)
                    pi = self._inner_ot(g_xs_mb, g_xt_mb, ys, pred_xt, total_cost,
                                        method, yt=yt_mb)

                    da_loss = 1.0 / k * torch.sum(pi * total_cost)
                    total_loss += da_loss
                    total_loss.backward()

                optimizer_g.step()
                optimizer_f.step()

            if id_epoch % self.test_interval == 0 or id_epoch == n_epochs - 1:
                source_acc = self.evaluate(source_loader)
                target_acc = self.evaluate(test_loader)
                self.logger.info(
                    "At epoch {} source and test accuracies are {} and {}".format(
                        id_epoch, source_acc, target_acc
                    )
                )
                save_acc(self.out_file, id_epoch, target_acc)
                if target_acc > best_acc:
                    best_acc = target_acc
                    torch.save(
                        {"model_g": self.model_g.state_dict(),
                         "model_f": self.model_f.state_dict(),
                         "epoch": id_epoch, "accuracy": target_acc},
                        os.path.join(self.out_dir, "best_model.pth"),
                    )

        torch.save(
            {"model_g": self.model_g.state_dict(),
             "model_f": self.model_f.state_dict(),
             "epoch": n_epochs, "accuracy": target_acc},
            os.path.join(self.out_dir, "final_model.pth"),
        )

    # ----- fit_bomb: full outer-plan weighting --------------------------

    def fit_bomb(self, source_loader, target_loader, test_loader,
                 n_epochs, criterion=nn.CrossEntropyLoss(), lr=2e-4,
                 k=1, batch_size=25, method="jumbot"):
        if self.use_kpg:
            raise NotImplementedError(
                "fit_bomb / fit_bomb2 cannot be combined with --use_kpg: the "
                "hierarchical batch-of-mini-batches scheme is outside the Sec. IV "
                "formulation, whose estimator is the uniform average of Eq. (8).  "
                "Use fit() for the keypoint-guided method."
            )
        criterion = nn.CrossEntropyLoss()
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)
        best_acc = 0

        for id_epoch in range(n_epochs):
            print(f"Epoch: {id_epoch}")
            self.model_g.train()
            self.model_f.train()
            target_loader_iter = iter(target_loader)

            for _, data in tqdm(enumerate(source_loader)):
                xs_mb_all, ys_all = data
                try:
                    xt_mb_all, yt_all = next(target_loader_iter)
                except StopIteration:
                    xt_mb_all = None
                if xt_mb_all is None or len(xt_mb_all) != batch_size:
                    target_loader_iter = iter(target_loader)
                    xt_mb_all, yt_all = next(target_loader_iter)

                inds_xs = np.split(np.arange(xs_mb_all.shape[0]), k)
                inds_xt = np.split(np.arange(xt_mb_all.shape[0]), k)

                # Phase 1: compute k*k cost matrix (no grad)
                list_da_loss = []
                with torch.no_grad():
                    for i in range(k):
                        xs_mb = xs_mb_all[inds_xs[i]].cuda()
                        g_xs_mb = self.model_g(xs_mb)
                        ys = ys_all[inds_xs[i]].cuda()
                        for j in range(k):
                            xt_mb = xt_mb_all[inds_xt[j]].cuda()
                            yt_mb = yt_all[inds_xt[j]]
                            g_xt_mb = self.model_g(xt_mb)
                            f_g_xt_mb = self.model_f(g_xt_mb)
                            pred_xt = F.softmax(f_g_xt_mb, 1)

                            total_cost = self._compute_cost(g_xs_mb, g_xt_mb, ys, pred_xt)
                            pi = self._inner_ot(g_xs_mb, g_xt_mb, ys, pred_xt,
                                                total_cost, method, yt=yt_mb)
                            list_da_loss.append(torch.sum(pi * total_cost))

                    big_C = torch.stack(list_da_loss).view(k, k)
                    if self.batch_epsilon == 0:
                        plan = ot.emd([], [], big_C.detach().cpu().numpy())
                    else:
                        plan = ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=self.batch_epsilon)

                # Phase 2: re-forward with grad
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                for i in range(k):
                    for j in range(k):
                        total_loss = 0
                        xs_mb = xs_mb_all[inds_xs[i]].cuda()
                        g_xs_mb = self.model_g(xs_mb)
                        f_g_xs_mb = self.model_f(g_xs_mb)
                        ys = ys_all[inds_xs[i]].cuda()

                        s_loss = 1.0 / (k ** 2) * criterion(f_g_xs_mb, ys)
                        total_loss += s_loss

                        if plan[i, j] == 0:
                            total_loss.backward()
                            continue

                        xt_mb = xt_mb_all[inds_xt[j]].cuda()
                        yt_mb = yt_all[inds_xt[j]]
                        g_xt_mb = self.model_g(xt_mb)
                        f_g_xt_mb = self.model_f(g_xt_mb)
                        pred_xt = F.softmax(f_g_xt_mb, 1)

                        total_cost = self._compute_cost(g_xs_mb, g_xt_mb, ys, pred_xt)
                        pi = self._inner_ot(g_xs_mb, g_xt_mb, ys, pred_xt,
                                            total_cost, method, yt=yt_mb)

                        da_loss = plan[i, j] * torch.sum(pi * total_cost)
                        total_loss += da_loss
                        total_loss.backward()

                optimizer_g.step()
                optimizer_f.step()

            if id_epoch % self.test_interval == 0 or id_epoch == n_epochs - 1:
                source_acc = self.evaluate(source_loader)
                target_acc = self.evaluate(test_loader)
                self.logger.info(
                    "At epoch {} source and test accuracies are {} and {}".format(
                        id_epoch, source_acc, target_acc
                    )
                )
                save_acc(self.out_file, id_epoch, target_acc)
                if target_acc > best_acc:
                    best_acc = target_acc
                    torch.save(
                        {"model_g": self.model_g.state_dict(),
                         "model_f": self.model_f.state_dict(),
                         "epoch": id_epoch, "accuracy": target_acc},
                        os.path.join(self.out_dir, "best_model.pth"),
                    )

        torch.save(
            {"model_g": self.model_g.state_dict(),
             "model_f": self.model_f.state_dict(),
             "epoch": n_epochs, "accuracy": target_acc},
            os.path.join(self.out_dir, "final_model.pth"),
        )

    # ----- fit_bomb2: stable 1-to-1 matching variant --------------------

    def fit_bomb2(self, source_loader, target_loader, test_loader,
                  n_epochs, criterion=nn.CrossEntropyLoss(), lr=2e-4,
                  k=1, batch_size=25, method="jumbot"):
        if self.use_kpg:
            raise NotImplementedError(
                "fit_bomb / fit_bomb2 cannot be combined with --use_kpg: the "
                "hierarchical batch-of-mini-batches scheme is outside the Sec. IV "
                "formulation, whose estimator is the uniform average of Eq. (8).  "
                "Use fit() for the keypoint-guided method."
            )
        criterion = nn.CrossEntropyLoss()
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)
        best_acc = 0

        for id_epoch in range(n_epochs):
            print(f"Epoch: {id_epoch}")
            self.model_g.train()
            self.model_f.train()
            target_loader_iter = iter(target_loader)

            for _, data in tqdm(enumerate(source_loader)):
                xs_mb_all, ys_all = data
                try:
                    xt_mb_all, yt_all = next(target_loader_iter)
                except StopIteration:
                    xt_mb_all = None
                if xt_mb_all is None or len(xt_mb_all) != batch_size:
                    target_loader_iter = iter(target_loader)
                    xt_mb_all, yt_all = next(target_loader_iter)

                inds_xs = np.split(np.arange(xs_mb_all.shape[0]), k)
                inds_xt = np.split(np.arange(xt_mb_all.shape[0]), k)

                # Phase 1: k*k cost + outer OT (no grad)
                list_da_loss = []
                with torch.no_grad():
                    for i in range(k):
                        xs_mb = xs_mb_all[inds_xs[i]].cuda()
                        g_xs_mb = self.model_g(xs_mb)
                        ys = ys_all[inds_xs[i]].cuda()
                        for j in range(k):
                            xt_mb = xt_mb_all[inds_xt[j]].cuda()
                            yt_mb = yt_all[inds_xt[j]]
                            g_xt_mb = self.model_g(xt_mb)
                            f_g_xt_mb = self.model_f(g_xt_mb)
                            pred_xt = F.softmax(f_g_xt_mb, 1)

                            total_cost = self._compute_cost(g_xs_mb, g_xt_mb, ys, pred_xt)
                            pi = self._inner_ot(g_xs_mb, g_xt_mb, ys, pred_xt,
                                                total_cost, method, yt=yt_mb)
                            list_da_loss.append(torch.sum(pi * total_cost))

                    big_C = torch.stack(list_da_loss).view(k, k)
                    if self.batch_epsilon == 0:
                        plan = ot.emd([], [], big_C.detach().cpu().numpy())
                    else:
                        plan = ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=self.batch_epsilon)
                    mapping = np.argmax(plan, axis=1)

                # Phase 2: re-forward only matched pairs (with grad)
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                for i in range(k):
                    j = mapping[i]
                    total_loss = 0

                    xs_mb = xs_mb_all[inds_xs[i]].cuda()
                    g_xs_mb = self.model_g(xs_mb)
                    f_g_xs_mb = self.model_f(g_xs_mb)
                    ys = ys_all[inds_xs[i]].cuda()

                    s_loss = 1.0 / k * criterion(f_g_xs_mb, ys)
                    total_loss += s_loss

                    xt_mb = xt_mb_all[inds_xt[j]].cuda()
                    yt_mb = yt_all[inds_xt[j]]
                    g_xt_mb = self.model_g(xt_mb)
                    f_g_xt_mb = self.model_f(g_xt_mb)
                    pred_xt = F.softmax(f_g_xt_mb, 1)

                    total_cost = self._compute_cost(g_xs_mb, g_xt_mb, ys, pred_xt)
                    pi = self._inner_ot(g_xs_mb, g_xt_mb, ys, pred_xt,
                                        total_cost, method, yt=yt_mb)

                    da_loss = plan[i, j] * torch.sum(pi * total_cost)
                    total_loss += da_loss
                    total_loss.backward()

                optimizer_g.step()
                optimizer_f.step()

            if id_epoch % self.test_interval == 0 or id_epoch == n_epochs - 1:
                source_acc = self.evaluate(source_loader)
                target_acc = self.evaluate(test_loader)
                self.logger.info(
                    "At epoch {} source and test accuracies are {} and {}".format(
                        id_epoch, source_acc, target_acc
                    )
                )
                save_acc(self.out_file, id_epoch, target_acc)
                if target_acc > best_acc:
                    best_acc = target_acc
                    torch.save(
                        {"model_g": self.model_g.state_dict(),
                         "model_f": self.model_f.state_dict(),
                         "epoch": id_epoch, "accuracy": target_acc},
                        os.path.join(self.out_dir, "best_model.pth"),
                    )

        torch.save(
            {"model_g": self.model_g.state_dict(),
             "model_f": self.model_f.state_dict(),
             "epoch": n_epochs, "accuracy": target_acc},
            os.path.join(self.out_dir, "final_model.pth"),
        )

    # ----- source-only pre-training -------------------------------------

    def source_only(self, source_loader, criterion=nn.CrossEntropyLoss(), lr=2e-4):
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)
        for _ in tqdm(range(10)):
            self.model_g.train()
            self.model_f.train()
            for _, data in enumerate(source_loader):
                xs_mb, ys = data
                xs_mb, ys = xs_mb.cuda(), ys.cuda()
                g_xs_mb = self.model_g(xs_mb)
                f_g_xs_mb = self.model_f(g_xs_mb)
                s_loss = criterion(f_g_xs_mb, ys)
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()
                s_loss.backward()
                optimizer_g.step()
                optimizer_f.step()
        source_acc = self.evaluate(source_loader)
        self.logger.info("Source accuracy is {}".format(source_acc))

    def evaluate(self, data_loader):
        return model_eval(data_loader, self.model_g, self.model_f)
