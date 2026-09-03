"""
Gradient flows in 2D
====================

Let's showcase the properties of **kernel MMDs**, **Hausdorff**
and **Sinkhorn** divergences on a simple toy problem:
the registration of one blob onto another.
"""
##############################################
# Setup
# ---------------------
import random
from random import choices

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import imageio.v2 as imageio
from imageio.v2 import imread
from utils import cost_matrix


np.random.seed(1)
torch.manual_seed(1)
random.seed(1)


def solve_uot_original(C, a, b, tau1, tau2, solver="ECOS", verbose=False):
    X = cp.Variable((a.shape[0], b.shape[0]))

    sum_X = cp.sum(X)
    sum_rowX = cp.sum(X, axis=1)
    sum_colX = cp.sum(X, axis=0)

    cost = cp.sum(cp.multiply(X, C))
    kl_row = (
        -cp.sum(cp.entr(sum_rowX))
        - cp.sum(
            cp.multiply(
                sum_rowX,
                cp.log(
                    a.reshape(
                        -1,
                    )
                ),
            )
        )
        - sum_X
        + cp.sum(
            a.reshape(
                -1,
            )
        )
    )
    kl_col = (
        -cp.sum(cp.entr(sum_colX))
        - cp.sum(
            cp.multiply(
                sum_colX,
                cp.log(
                    b.reshape(
                        -1,
                    )
                ),
            )
        )
        - sum_X
        + cp.sum(
            b.reshape(
                -1,
            )
        )
    )

    total_cost = cost + tau1 * kl_row + tau2 * kl_col

    objective = cp.Minimize(total_cost)
    constraints = [0 <= X]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=solver, verbose=verbose)
        if prob.status in (None, "infeasible", "unbounded"):
            raise cp.error.SolverError("bad status")
    except cp.error.SolverError:
        prob.solve(solver="SCS", verbose=verbose)

    return prob.value, X.value


def compute_true_Wasserstein(X, Y, p=2):
    M = ot.dist(X.detach().numpy(), Y.detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)


def compute_Wasserstein(M, device="cuda", e=0):
    if e == 0:
        pi = ot.emd([], [], M.cpu().detach().numpy()).astype("float32")
    else:
        pi = ot.sinkhorn([], [], M.cpu().detach().numpy(), reg=e).astype("float32")
    pi = torch.from_numpy(pi).to(M.device)
    return torch.sum(pi * M)


def mOT(firsttensor, secondtensor, p=2, device="cpu", numbatch=4, batch_size=16, e=0):
    inds1 = []
    inds2 = []
    for _ in range(numbatch):
        inds1.append(np.random.choice(firsttensor.shape[0], batch_size, replace=False))
        inds2.append(np.random.choice(firsttensor.shape[0], batch_size, replace=False))
    ll = []
    for i in range(numbatch):
        for j in range(numbatch):
            M = cost_matrix(firsttensor[inds1[i]], secondtensor[inds2[j]], p)
            w = compute_Wasserstein(M, device, e)
            ll.append(w)
    return torch.stack(ll).mean()


def BoMbOT(firsttensor, secondtensor, p=2, device="cpu", numbatch=4, batch_size=16, e=0):
    inds1 = []
    inds2 = []
    for _ in range(numbatch):
        inds1.append(np.random.choice(firsttensor.shape[0], batch_size, replace=False))
        inds2.append(np.random.choice(firsttensor.shape[0], batch_size, replace=False))
    ll = []
    for i in range(numbatch):
        for j in range(numbatch):
            M = cost_matrix(firsttensor[inds1[i]], secondtensor[inds2[j]], p)
            w = compute_Wasserstein(M, device, e=e)
            ll.append(w)
    M = torch.stack(ll).view(numbatch, numbatch)
    return compute_Wasserstein(M, device)


def eBoMbOT(firsttensor, secondtensor, p=2, breg=0.01, device="cpu", numbatch=4, batch_size=16, e=0):
    inds1 = []
    inds2 = []
    for _ in range(numbatch):
        inds1.append(np.random.choice(firsttensor.shape[0], batch_size, replace=False))
        inds2.append(np.random.choice(firsttensor.shape[0], batch_size, replace=False))
    ll = []
    for i in range(numbatch):
        for j in range(numbatch):
            M = cost_matrix(firsttensor[inds1[i]], secondtensor[inds2[j]], p)
            w = compute_Wasserstein(M, device, e)
            ll.append(w)
    M = torch.stack(ll).view(numbatch, numbatch)
    return compute_Wasserstein(M, device, e=breg)


def mUOT(firsttensor, secondtensor, p=2, device="cpu", numbatch=4, batch_size=4, reg=1, tau=0.01):
    inds1 = []
    inds2 = []
    for _ in range(numbatch):
        inds1.append(np.random.choice(firsttensor.shape[0], batch_size, replace=False))
        inds2.append(np.random.choice(firsttensor.shape[0], batch_size, replace=False))
    ll = []
    for i in range(numbatch):
        for j in range(numbatch):
            M = cost_matrix(firsttensor[inds1[i]], secondtensor[inds2[j]], p)
            _, pi = solve_uot_original(
                M.cpu().detach().numpy().astype("float32"),
                np.ones(batch_size) / batch_size,
                np.ones(batch_size) / batch_size,
                tau,
                tau,
            )
            pi = torch.from_numpy(pi).to(M.device)
            w = torch.sum(pi * M)
            ll.append(w)
    return torch.stack(ll).mean()


def BoMbUOT(firsttensor, secondtensor, p=2, device="cuda", numbatch=4, batch_size=4, reg=1, tau=0.01):
    inds1 = []
    inds2 = []
    for _ in range(numbatch):
        inds1.append(np.random.choice(firsttensor.shape[0], batch_size, replace=False))
        inds2.append(np.random.choice(firsttensor.shape[0], batch_size, replace=False))
    ll = []
    for i in range(numbatch):
        for j in range(numbatch):
            M = cost_matrix(firsttensor[inds1[i]], secondtensor[inds2[j]], p)
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(
                np.ones(batch_size) / batch_size,
                np.ones(batch_size) / batch_size,
                M.cpu().detach().numpy().astype("float32"),
                reg=reg,
                reg_m=tau,
            )
            pi = torch.from_numpy(pi).to(M.device)
            w = torch.sum(pi * M)
            ll.append(w)
    M = torch.stack(ll).view(numbatch, numbatch)
    return compute_Wasserstein(M, device)


def mPOT(firsttensor, secondtensor, p=2, device="cuda", numbatch=4, batch_size=4, mass=0.8, e=1):
    inds1 = []
    inds2 = []
    for _ in range(numbatch):
        inds1.append(np.random.choice(firsttensor.shape[0], batch_size, replace=False))
        inds2.append(np.random.choice(firsttensor.shape[0], batch_size, replace=False))
    ll = []
    for i in range(numbatch):
        for j in range(numbatch):
            M = cost_matrix(firsttensor[inds1[i]], secondtensor[inds2[j]], p)
            if e == 0:
                pi = ot.partial.partial_wasserstein(
                    np.ones(batch_size) / batch_size,
                    np.ones(batch_size) / batch_size,
                    M.cpu().detach().numpy().astype("float32"),
                    m=mass,
                )
            else:
                pi = ot.partial.entropic_partial_wasserstein(
                    np.ones(batch_size) / batch_size,
                    np.ones(batch_size) / batch_size,
                    M.cpu().detach().numpy().astype("float32"),
                    m=mass,
                    reg=e,
                )
            pi = torch.from_numpy(pi).to(M.device)
            w = torch.sum(pi * M)
            ll.append(w)
    return torch.stack(ll).mean()


def BoMbPOT(firsttensor, secondtensor, p=2, device="cuda", numbatch=4, batch_size=4, mass=0.8, e=1):
    inds1 = []
    inds2 = []
    for _ in range(numbatch):
        inds1.append(np.random.choice(firsttensor.shape[0], batch_size, replace=False))
        inds2.append(np.random.choice(firsttensor.shape[0], batch_size, replace=False))
    ll = []
    for i in range(numbatch):
        for j in range(numbatch):
            M = cost_matrix(firsttensor[inds1[i]], secondtensor[inds2[j]], p)
            if e == 0:
                pi = ot.partial.partial_wasserstein(
                    np.ones(batch_size) / batch_size,
                    np.ones(batch_size) / batch_size,
                    M.cpu().detach().numpy().astype("float32"),
                    m=mass,
                )
            else:
                pi = ot.partial.entropic_partial_wasserstein(
                    np.ones(batch_size) / batch_size,
                    np.ones(batch_size) / batch_size,
                    M.cpu().detach().numpy().astype("float32"),
                    m=mass,
                    reg=e,
                )
            pi = torch.from_numpy(pi).to(M.device)
            w = torch.sum(pi * M)
            ll.append(w)
    M = torch.stack(ll).view(numbatch, numbatch)
    return compute_Wasserstein(M, device)


use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

###############################################
# Display routine
# ~~~~~~~~~~~~~~~~~


def load_image(fname):
    img = imread(fname, mode='F')  # Grayscale
    img = (img[::-1, :]) / 255.0
    return 1 - img


def draw_samples(fname, n, dtype=torch.FloatTensor):
    A = load_image(fname)
    xg, yg = np.meshgrid(np.linspace(0, 1, A.shape[0]), np.linspace(0, 1, A.shape[1]))

    grid = list(zip(xg.ravel(), yg.ravel()))
    dens = A.ravel() / A.sum()
    dots = np.array(choices(grid, dens, k=n))
    dots += (0.5 / A.shape[0]) * np.random.standard_normal(dots.shape)

    return torch.from_numpy(dots).type(dtype)


def display_samples(ax, x, color):
    x_ = x.detach().cpu().numpy()
    ax.scatter(x_[:, 0], x_[:, 1], 25 * 500 / len(x_), color, edgecolors="none")


np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
N, M = (1000, 1000) if not use_cuda else (1000, 1000)

X_i = draw_samples("data/density_a.png", N, dtype)
Y_j = draw_samples("data/density_b.png", M, dtype)


def gradient_flow(loss, axes, lr=0.001, title="m-OT", flag=False):
    """Flows along the gradient of the cost function, using a simple Euler scheme.

    Parameters:
        loss ((x_i,y_j) -> torch float number):
            Real-valued loss function.
        axes (list of 6 matplotlib Axes):
            Pre-created axes for the 6 time snapshots.
        lr (float, default = .05):
            Learning rate, i.e. time step.
    """

    # Parameters for the gradient descent
    Nsteps = int(5 / lr) + 1
    display_its = [int(t / lr) for t in [0, 1, 2, 3, 4, 5.0]]

    # Use colors to identify the particles
    colors = (10 * X_i[:, 0]).cos() * (10 * X_i[:, 1]).cos()
    colors = colors.detach().cpu().numpy()

    # Make sure that we won't modify the reference samples
    x_i, y_j = X_i.clone(), Y_j.clone()

    # We're going to perform gradient descent on Loss(α, β)
    # wrt. the positions x_i of the diracs masses that make up α:
    x_i.requires_grad = True

    k = 0
    for i in range(Nsteps):  # Euler scheme ===============
        # Compute cost and gradient
        L_αβ = loss(x_i, y_j)
        [g] = torch.autograd.grad(L_αβ, [x_i])

        if i in display_its:  # display
            ax = axes[k]
            if k == 0:
                ax.set_ylabel(title, fontsize=11)
            ax.scatter([10], [10])  # shameless hack to prevent a slight change of axis...

            display_samples(ax, y_j, [(0.55, 0.55, 0.95)])
            display_samples(ax, x_i, colors)

            ax.set_title(
                "$W_2$: "
                + str(np.round(compute_true_Wasserstein(x_i.cpu(), y_j.cpu()) * 100, 4))
                + r"$\times 10^{-2}$",
                fontsize=11,
            )
            if flag:
                ax.set_xlabel("steps " + str(i), fontsize=11)
            ax.axis([0, 1, 0, 1])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            k += 1

        # in-place modification of the tensor's values
        x_i.data -= lr * len(x_i) * g


plt.set_cmap("hsv")
fig, all_axes = plt.subplots(4, 6, figsize=(18, 12))

methods = [
    (mOT,                                   "m-OT",                                   False),
    (mUOT,                                  r"m-UOT $\epsilon=1, \tau=10^{-2}$",      False),
    (lambda x, y: mPOT(x, y, mass=0.9),    "m-POT s=0.9",                            True),
    (mPOT,                                  "m-POT s=0.8",                            True),
]

for row_idx, (method, title, flag) in enumerate(methods):
    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)
    gradient_flow(method, all_axes[row_idx], title=title, flag=flag)

plt.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.95, wspace=0.05, hspace=0.3)
plt.savefig("Gradient_Flow.png", dpi=150, bbox_inches="tight")
plt.show()
