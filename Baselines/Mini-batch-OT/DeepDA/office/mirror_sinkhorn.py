"""
Mirror Sinkhorn inner solver for m-OT / m-UOT / m-POT.

Drop-in replacement for the entropic-Sinkhorn / EMD inner solves used by
the mini-batch-OT pipelines (Nguyen et al., ICML 2022 — the baseline this
file lives in).  Provides three entry points covering the three OT types:

    mirror_sinkhorn_balanced(a, b, C, n_iter)            ⇒ m-OT
    mirror_sinkhorn_unbalanced(a, b, C, tau, n_iter)     ⇒ m-UOT (L1 variant)
    mirror_sinkhorn_partial(a, b, C, mass, n_iter)       ⇒ m-POT

and a dispatch helper `mirror_sinkhorn(ot_type, ...)` so call sites only
have to switch on a single string.

Algorithm
---------
Algorithm 1 of Ballu & Berthet, "Mirror Sinkhorn: Fast Online Optimization
on Transport Polytopes", ICML 2023.  For a linear objective ⟨C, γ⟩:

    γ_1 = a b^T
    for t = 1 .. T:
        γ'_{t+1} = γ_t ⊙ exp(-η_t · C)              # mirror-descent step
        γ_{t+1}  = renormalise row t-even / col t-odd to a, b
        γ̄_{t+1} = (t γ̄_t + γ_{t+1}) / (t + 1)         # Cesaro average
    return γ̄_T

Step size η_t = (1/B) √(δ/t)  with B = ‖C‖_∞ and δ = ‖log a‖_∞ + ‖log b‖_∞
(general B-Lipschitz form, Thm A.1; for normalized cost ‖C‖_∞=1 this is the
OT corollary Thm 3.3 with σ=0).  No per-experiment tuning.

Exact marginals (optional).  Mirror Sinkhorn's iterates satisfy the marginals
only polynomially (γ̄_T ∉ T(a,b) exactly).  Pass `round_output=True` to apply
the paper's Algorithm 2 rounding (Altschuler et al., 2017) and obtain a plan
with row sums = a and column sums = b EXACTLY.  This is only meaningful for the
balanced type; the unbalanced / partial variants relax the marginals by
construction and reject the flag.

Why this vs entropic Sinkhorn
-----------------------------
  • Bias-free: ⟨C, γ̄_T⟩ → ⟨C, γ*⟩ on the *true* OT, not the α-regularised
    optimum that entropic Sinkhorn plateaus at.  Useful when the OT loss
    is feeding a neural-net gradient.
  • Stable: no exp(-cost/reg) underflow, no NaN at small mass.
  • Online-friendly: warm-starting γ across mini-batches is a one-line
    change (not used here, but available).

Unbalanced note
---------------
The paper handles the *transport polytope* (hard marginals).  Adding KL
marginal regularisers does not change the iterates (Prop 2.1), so KL-UOT
isn't reachable by Mirror Sinkhorn directly.  We provide an L1-style
unbalanced variant via slack augmentation: extra slack rows / cols absorb
marginal mismatch at unit cost τ.  This is a related but distinct convex
relaxation from `ot.unbalanced.sinkhorn_knopp_unbalanced` — tune τ to
match the desired marginal-strictness.
"""

import numpy as np


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _safe_log(x, eps=1e-20):
    return np.log(np.maximum(x, eps))


def _step_size(t, B, delta):
    return (1.0 / max(B, 1e-10)) * np.sqrt(delta / max(t, 1))


def _round_transport(gamma, a, b):
    """Round a non-negative matrix onto the transport polytope T(a, b).

    Algorithm 2 of Altschuler, Niles-Weed & Rigollet (NeurIPS 2017), exactly
    as invoked by the Mirror Sinkhorn paper (Ballu & Berthet, ICML 2023, Alg. 2
    / Prop. A.5).  Returns a matrix whose row sums equal `a` and column sums
    equal `b` EXACTLY (up to floating-point), with
        ||round(gamma) - gamma||_1 <= 2 * ( ||gamma 1 - a||_1 + ||gamma^T 1 - b||_1 ).

    Steps (entrywise minimum denoted by min(.,1)):
      1. rows:  gamma'  = Diag( min(a / (gamma 1), 1) )  gamma
      2. cols:  gamma'' = gamma' Diag( min(b / (gamma'^T 1), 1) )
      3. rank-1 completion of the residual deficits:
                rho = gamma'' + (a - gamma'' 1)(b - gamma''^T 1)^T / ||a - gamma'' 1||_1
    """
    gamma = np.asarray(gamma, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # Step 1 — shrink any row that overshoots its target a_i (rows only scale down).
    row = gamma.sum(axis=1)
    x = np.minimum(a / np.maximum(row, 1e-300), 1.0)
    g = gamma * x[:, None]

    # Step 2 — shrink any column that overshoots its target b_j.
    col = g.sum(axis=0)
    y = np.minimum(b / np.maximum(col, 1e-300), 1.0)
    g = g * y[None, :]

    # Step 3 — rank-1 correction with the leftover deficits (both >= 0, equal sum).
    err_r = a - g.sum(axis=1)
    err_c = b - g.sum(axis=0)
    denom = err_r.sum()                 # == err_c.sum() == leftover mass
    if denom > 1e-300:
        g = g + np.outer(err_r, err_c) / denom
    return g


# -----------------------------------------------------------------------
# Balanced — m-OT
# -----------------------------------------------------------------------

def mirror_sinkhorn_balanced(
    a, b, C, n_iter=500, return_avg=True, normalize_cost=True, forbidden=None,
    round_output=False,
):
    """Mirror Sinkhorn for balanced OT.

    Solves  min_{γ ∈ T(a, b)}  ⟨C, γ⟩.

    Parameters
    ----------
    a, b           : 1-D ndarrays of equal sum (must satisfy sum(a)=sum(b)).
    C              : (m, n) cost matrix.
    n_iter         : iterations.  500 is a generous default for batches of
                     ~32–500; reduce for speed.
    return_avg     : True ⇒ Cesaro average γ̄_T (bias-free, Thm 3.1).
                     False ⇒ last iterate γ_T (sparser but biased).
    normalize_cost : divide C by max|C| so the step-size formula's
                     B = 1 assumption holds.  Keep True unless you've
                     scaled C yourself.
    forbidden      : optional (m, n) bool ndarray.  Entries with True are
                     zeroed in the initial iterate and stay zero forever
                     (the multiplicative update preserves zeros).  Used
                     internally by the partial-OT variant to forbid the
                     slack-to-slack corner.
    round_output   : if True, apply the paper's Algorithm 2 rounding
                     (`_round_transport`) so the returned plan has row sums
                     EXACTLY `a` and column sums EXACTLY `b`.  Mirror Sinkhorn's
                     marginals otherwise converge only polynomially, so this is
                     the way to obtain an exactly-feasible plan.  Not compatible
                     with `forbidden` (the rank-1 correction would repopulate the
                     forbidden cell) — raises ValueError in that case.

    Returns
    -------
    pi : (m, n) ndarray, the transport plan.  Marginals match a, b up to
        an error that decays as O(√(δ / n_iter)) — or EXACTLY if round_output.
    """
    if round_output and forbidden is not None:
        raise ValueError(
            "round_output=True is incompatible with a forbidden mask: "
            "Algorithm 2's rank-1 correction would put mass back into the "
            "forbidden cell.  Partial OT has relaxed marginals by construction, "
            "so exact-marginal rounding does not apply."
        )
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    if normalize_cost:
        scale = max(np.max(np.abs(C)), 1e-10)
        C = C / scale
    B = max(np.max(np.abs(C)), 1e-10)
    delta = float(np.max(np.abs(_safe_log(a))) + np.max(np.abs(_safe_log(b))))

    gamma = np.outer(a, b)
    if forbidden is not None:
        gamma[forbidden] = 0.0
        # Few Sinkhorn projections to restore marginals after masking.
        # The multiplicative update afterwards preserves zeros forever.
        for _ in range(20):
            row_sums = np.maximum(gamma.sum(axis=1), 1e-20)
            gamma = gamma * (a / row_sums)[:, None]
            col_sums = np.maximum(gamma.sum(axis=0), 1e-20)
            gamma = gamma * (b / col_sums)[None, :]
    gamma_bar = gamma.copy()

    for t in range(1, n_iter + 1):
        eta = _step_size(t, B, delta)
        gamma_prime = gamma * np.exp(-eta * C)

        if t % 2 == 0:
            row_sums = np.maximum(gamma_prime.sum(axis=1), 1e-20)
            gamma = gamma_prime * (a / row_sums)[:, None]
        else:
            col_sums = np.maximum(gamma_prime.sum(axis=0), 1e-20)
            gamma = gamma_prime * (b / col_sums)[None, :]

        gamma_bar = (t * gamma_bar + gamma) / (t + 1)

    out = gamma_bar if return_avg else gamma
    if round_output:
        out = _round_transport(out, a, b)     # exact marginals (Algorithm 2)
    return out


# -----------------------------------------------------------------------
# Unbalanced — m-UOT (L1-marginal variant via slack augmentation)
# -----------------------------------------------------------------------

def mirror_sinkhorn_unbalanced(
    a, b, C, tau, n_iter=500, return_avg=True, normalize_cost=True,
    round_output=False,
):
    """Mirror Sinkhorn for L1-style unbalanced OT.

    Solves
        min_{γ ≥ 0}  ⟨C, γ⟩ + τ ‖γ 1 − a‖_1 + τ ‖γ^T 1 − b‖_1
    via slack augmentation:
        a' = [a; sum(b)],   b' = [b; sum(a)]
        C_aug[:m, :n] = C ;  C_aug slack row/col = τ ;  C_aug slack corner = 0.
    Returns the real (m × n) sub-block of γ̄.

    This is a different relaxation than the KL-UOT solved by
    `ot.unbalanced.sinkhorn_knopp_unbalanced` (which uses KL marginal
    divergences).  Adjust τ for comparable marginal-strictness.

    `round_output` is rejected: the returned sub-block has *relaxed* marginals
    by design, so there is no fixed (a, b) target for Algorithm 2 to round to.
    """
    if round_output:
        raise ValueError(
            "round_output is only defined for balanced OT.  Unbalanced OT "
            "deliberately relaxes the marginals, so there is no exact (a, b) "
            "target for the paper's Algorithm 2 rounding."
        )
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    m, n = C.shape

    sum_a, sum_b = float(a.sum()), float(b.sum())
    a_aug = np.concatenate([a, [sum_b]])
    b_aug = np.concatenate([b, [sum_a]])

    if normalize_cost:
        scale = max(np.max(np.abs(C)), 1e-10)
        C = C / scale
        tau = float(tau) / scale

    C_aug = np.zeros((m + 1, n + 1), dtype=np.float64)
    C_aug[:m, :n] = C
    C_aug[:m, n]  = tau
    C_aug[m, :n]  = tau
    C_aug[m, n]   = 0.0

    pi_aug = mirror_sinkhorn_balanced(
        a_aug, b_aug, C_aug, n_iter=n_iter,
        return_avg=return_avg, normalize_cost=False,
    )
    return pi_aug[:m, :n]


# -----------------------------------------------------------------------
# Partial — m-POT (slack augmentation)
# -----------------------------------------------------------------------

def mirror_sinkhorn_partial(
    a, b, C, mass, n_iter=500, return_avg=True, normalize_cost=True,
    round_output=False,
):
    """Mirror Sinkhorn for partial OT.

    Solves   min_{γ ≥ 0}  ⟨C, γ⟩  s.t.  γ 1 ≤ a, γ^T 1 ≤ b, sum(γ) = mass.

    Embedding:
        a' = [a;  sum(b) − mass],   b' = [b;  sum(a) − mass]
        C_aug[:m, :n] = C ;  slack rows / cols are cost-free.
        slack-to-slack corner is forbidden via the `forbidden` mask of
        `mirror_sinkhorn_balanced` (zeroed in init, stays zero forever),
        which exactly enforces sum(real block) = mass.

    `round_output` is rejected: partial OT has inequality (relaxed) marginals
    and a forbidden slack-to-slack corner, so the balanced-polytope Algorithm 2
    rounding does not apply.
    """
    if round_output:
        raise ValueError(
            "round_output is only defined for balanced OT.  Partial OT uses "
            "inequality marginals (γ1 ≤ a, γ^T 1 ≤ b) and a forbidden slack "
            "corner, so the paper's Algorithm 2 (balanced-polytope rounding) "
            "does not apply."
        )
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    m, n = C.shape

    sum_a, sum_b = float(a.sum()), float(b.sum())
    if mass > min(sum_a, sum_b) + 1e-9:
        raise ValueError(
            f"Infeasible partial OT: mass={mass} > min(sum(a)={sum_a}, "
            f"sum(b)={sum_b})."
        )
    slack_src = max(sum_b - mass, 0.0)
    slack_tgt = max(sum_a - mass, 0.0)

    a_aug = np.concatenate([a, [slack_src]])
    b_aug = np.concatenate([b, [slack_tgt]])

    if normalize_cost:
        scale = max(np.max(np.abs(C)), 1e-10)
        C = C / scale

    C_aug = np.zeros((m + 1, n + 1), dtype=np.float64)
    C_aug[:m, :n] = C

    forbidden = np.zeros((m + 1, n + 1), dtype=bool)
    forbidden[m, n] = True   # slack → slack ⇒ sum(real block) = mass exactly

    pi_aug = mirror_sinkhorn_balanced(
        a_aug, b_aug, C_aug, n_iter=n_iter, return_avg=return_avg,
        normalize_cost=False, forbidden=forbidden,
    )
    return pi_aug[:m, :n]


# -----------------------------------------------------------------------
# Dispatch — single string switch for call sites
# -----------------------------------------------------------------------

def mirror_sinkhorn(
    ot_type, a, b, C, *, mass=None, tau=None,
    n_iter=500, return_avg=True, normalize_cost=True, round_output=False,
):
    """Pick the right Mirror Sinkhorn variant for `ot_type`.

    Accepts the naming conventions used across the baselines:
        balanced  / ot   / jdot     ⇒ mirror_sinkhorn_balanced
        unbalanced / uot / jumbot   ⇒ mirror_sinkhorn_unbalanced (needs tau)
        partial   / pot  / jpmbot   ⇒ mirror_sinkhorn_partial    (needs mass)

    `round_output=True` applies the paper's Algorithm 2 rounding for EXACT
    marginals.  It is only valid for the balanced types; unbalanced/partial
    raise (their marginals are relaxed by construction).
    """
    key = ot_type.lower()
    if key in ("ot", "balanced", "jdot"):
        return mirror_sinkhorn_balanced(
            a, b, C, n_iter=n_iter, return_avg=return_avg,
            normalize_cost=normalize_cost, round_output=round_output,
        )
    if key in ("uot", "unbalanced", "jumbot"):
        if tau is None:
            raise ValueError("Mirror Sinkhorn unbalanced needs `tau`.")
        return mirror_sinkhorn_unbalanced(
            a, b, C, tau=tau, n_iter=n_iter, return_avg=return_avg,
            normalize_cost=normalize_cost, round_output=round_output,
        )
    if key in ("pot", "partial", "jpmbot"):
        if mass is None:
            raise ValueError("Mirror Sinkhorn partial needs `mass`.")
        return mirror_sinkhorn_partial(
            a, b, C, mass=mass, n_iter=n_iter, return_avg=return_avg,
            normalize_cost=normalize_cost, round_output=round_output,
        )
    raise ValueError(f"Unknown ot_type for Mirror Sinkhorn: {ot_type}")
