"""
Keypoint-guided mini-batch OT for the Deep Generative Models experiments (Sec. IV).

The Sec. IV core below is shared verbatim with the three domain-adaptation drivers; this
module holds the NumPy-only copy that `Celeba_generator.py` and `Cifar_generator.py`
import, plus the one piece the generative setting needs on top of it:

  - Keypoint mining.  DeepGM is unsupervised, so Sec. IV-A's annotated pairs are not
    available.  As Sec. V-E describes, we solve the UNGUIDED problem once and keep its
    highest-mass correspondences (`_select_keypoint_pairs_from_plan`).  A generative
    keypoint pair is (a fixed real image, a fixed latent code z): the pairs stay fixed for
    the whole run while the generated member's features move with the generator, the exact
    analogue of the DA keypoints' embeddings moving with the encoder.

Everything else is the paper's formulation as written: the k pairs occupy the first k slots
of every mini-batch (Eq. 5), the mask pins them to their partners (Eqs. 6-7), and the
blended cost (Eq. 10) is solved on the masked feasible set.  `solve_kuot_paper` is the
entry point.  There is no `kp_strategy` here — the keypoints are plan-mined, not chosen by
one of the DA selection rules.
"""

import numpy as np
import ot

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

# Sec. IV formulation (mask + fixed keypoints + derived scaling solver).


def solve_kuot_paper(C_np, feat_real, feat_fake, k, alpha=0.5, rho=0.1,
                     metric="euclidean", method="UOT", reg=0.01, tau=1.0, mass=1.0,
                     n_iter=1000):
    """Sec. IV formulation for the generative setting.

    Assumes the batch has already been assembled with the k fixed keypoint pairs in its
    first k slots (Eq. (5)) — for DeepGM a keypoint pair is (a fixed real image, a fixed
    latent code z), so the generated member's features move as the generator trains, exactly
    as the DA keypoints' embeddings move as the encoder trains.

    Builds the mask (Eq. 6), the guiding matrix (Eq. 9) and the blended cost (Eq. 10), then
    solves the masked problem — by the derived scaling algorithm (Eq. 15) for `UOT`.
    """
    m = C_np.shape[0]
    M = build_mask(m, k)
    G = guiding_matrix(
        np.asarray(feat_real, dtype=np.float64),
        np.asarray(feat_fake, dtype=np.float64),
        k, rho=rho, metric=metric,
    )
    C_tilde = blend_cost(np.asarray(C_np, dtype=np.float64), G, alpha)
    ot_type = {"OT": "balanced", "UOT": "unbalanced", "POT": "partial"}[method]
    # Only the unbalanced solver requires reg > 0.  The generative experiments run
    # `--method OT --reg 0`, i.e. the BALANCED regime with the exact network-simplex
    # solver: falling back to 0.01 there would silently turn m-KOT into entropic OT.
    eps_eff = reg
    if ot_type == "unbalanced" and eps_eff <= 0:
        eps_eff = 0.01
    return solve_kuot(C_tilde, M, ot_type=ot_type, eps=eps_eff,
                           tau=tau, mass=mass, n_iter=n_iter)


# -----------------------------------------------------------------------
# Keypoint mining (DeepGM only — no labels, so Sec. V-E's plan-mined pairs)
# -----------------------------------------------------------------------

def _select_keypoint_pairs_from_plan(pi_init, n_kp):
    """Greedily pick the top-n_kp (row, col) pairs from pi_init.

    Each row / col can appear at most once.  Returns (I_kp, J_kp) — the real / fake
    sample indices of the k keypoint pairs the relation profiles are built against.
    """
    flat = pi_init.flatten()
    order = np.argsort(-flat)
    used_rows, used_cols = set(), set()
    I_kp, J_kp = [], []
    for idx in order:
        if len(I_kp) >= n_kp:
            break
        i, j = divmod(int(idx), pi_init.shape[1])
        if i in used_rows or j in used_cols:
            continue
        I_kp.append(i)
        J_kp.append(j)
        used_rows.add(i)
        used_cols.add(j)
    return I_kp, J_kp


# -----------------------------------------------------------------------
# Unguided OT solver — the m-OT / m-UOT / m-POT baselines, and the initial
# plan the keypoints above are mined from
# -----------------------------------------------------------------------

def solve_ot(a, b, C_np, method, reg, tau, mass):
    if method == "OT":
        if reg == 0:
            return ot.emd(a, b, C_np)
        else:
            return ot.sinkhorn(a, b, C_np, reg=reg)
    elif method == "UOT":
        return ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, C_np, reg=reg, reg_m=tau)
    elif method == "POT":
        if reg == 0:
            return ot.partial.partial_wasserstein(a, b, C_np, m=mass)
        else:
            return ot.partial.entropic_partial_wasserstein(a, b, C_np, m=mass, reg=reg)

