from ot.utils import list_to_array
from ot.backend import get_backend
from ot.lp import emd
import numpy as np
import warnings
import pdb

def entropic_partial_wasserstein_logscale(
    a, b, M, reg, m=None, numItermax=1000, stopThr=1e-100, verbose=False, log=False
):
    r"""
    Solves the partial optimal transport problem
    and returns the OT plan

    This function should return the same result as the function
    :any:`entropic_partial_wasserstein`, but is more stable and slower.

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma,
                 \mathbf{M} \rangle_F + \mathrm{reg} \cdot\Omega(\gamma)

        s.t. \gamma \mathbf{1} &\leq \mathbf{a} \\
             \gamma^T \mathbf{1} &\leq \mathbf{b} \\
             \gamma &\geq 0 \\
             \mathbf{1}^T \gamma^T \mathbf{1} = m
             &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\} \\

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\Omega`  is the entropic regularization term,
      :math:`\Omega=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights
    - `m` is the amount of mass to be transported

    The formulation of the problem has been proposed in
    :ref:`[3] <references-entropic-partial-wasserstein>` (prop. 5)


    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : np.ndarray (dim_b,)
        Unnormalized histograms of dimension `dim_b`
    M : np.ndarray (dim_a, dim_b)
        cost matrix
    reg : float
        Regularization term > 0
    m : float, optional
        Amount of mass to be transported
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (dim_a, dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> a = [.1, .2]
    >>> b = [.1, .1]
    >>> M = [[0., 1.], [2., 3.]]
    >>> np.round(entropic_partial_wasserstein_logscale(a, b, M, 1, 0.1), 2)
    array([[0.06, 0.02],
           [0.01, 0.  ]])


    .. _references-entropic-partial-wasserstein:
    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
       (2015). Iterative Bregman projections for regularized transportation
       problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    See Also
    --------
    ot.partial.partial_wasserstein: exact Partial Wasserstein
    """

    a, b, M = list_to_array(a, b, M)

    nx = get_backend(a, b, M)

    dim_a, dim_b = M.shape

    Ldx = nx.zeros(dim_a, type_as=a)
    Ldy = nx.zeros(dim_b, type_as=b)

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=a) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=b) / dim_b

    La = nx.log(a)
    Lb = nx.log(b)

    if m is None:
        m = nx.min(nx.stack((nx.sum(a), nx.sum(b)))) * 1.0
    if m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater" " than 0.")
    if m > nx.min(nx.stack((nx.sum(a), nx.sum(b)))):
        raise ValueError(
            "Problem infeasible. Parameter m should lower or"
            " equal than min(|a|_1, |b|_1)."
        )

    log_e = {"err": []}

    LK = -M / reg
    LK = LK + nx.log(m) - nx.logsumexp(LK)

    err, cpt = 1, 0

    Lq1 = nx.zeros(M.shape, type_as=M)
    Lq2 = nx.zeros(M.shape, type_as=M)  
    Lq3 = nx.zeros(M.shape, type_as=M)

    while err > stopThr and cpt < numItermax:
        LKprev = LK
        LK = LK + Lq1
        LK1 = nx.reshape(nx.minimum(La - nx.logsumexp(LK, 1), Ldx), (-1, 1)) + LK
        Lq1 = Lq1 + LKprev - LK1
        LK1prev = LK1
        LK1 = LK1 + Lq2
        LK2 =  LK1 + nx.reshape(nx.minimum(Lb - nx.logsumexp(LK1, 0), Ldy), (1, -1))
        Lq2 = Lq2 + LK1prev - LK2
        LK2prev = LK2
        LK2 = LK2 + Lq3
        LK = LK2 + nx.log(m) - nx.logsumexp(LK2)
        Lq3 = Lq3 + LK2prev - LK

        if nx.any(nx.isnan(LK)) or nx.any(nx.isinf(LK)):
            print("Warning: numerical errors at iteration", cpt)
            break
        if cpt % 10 == 0:
            err = nx.norm(LKprev - LK)
            if log:
                log_e["err"].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 11)
                print("{:5d}|{:8e}|".format(cpt, err))

        cpt = cpt + 1
    log_e["partial_w_dist"] = nx.sum(M * nx.exp(LK))
    if log:
        return nx.exp(LK), log_e
    else:
        return nx.exp(LK)