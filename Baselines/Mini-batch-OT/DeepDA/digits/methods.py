"""
Deep Domain Adaptation with Mini-batch Optimal Transport (mOT) and
Batch of Mini-batches OT (BoMb-OT).

Implements three training strategies for unsupervised domain adaptation:
  1. fit()       — Standard mOT: average OT losses across k independent mini-batch pairs.
  2. fit_bomb()  — BoMb-OT: solve a k×k outer OT to optimally match source/target mini-batches,
                   then weight each pair's loss by the outer plan.
  3. fit_bomb2() — Stable BoMb-OT variant: use argmax on the outer plan to get a
                   deterministic 1-to-1 matching (avoids numerical issues when losses are small).

Each strategy supports three inner OT solvers:
  - "jdot"   : standard OT  (ot.emd / ot.sinkhorn)
  - "jumbot" : unbalanced OT (ot.unbalanced.sinkhorn_knopp_unbalanced)
  - "jpmbot" : partial OT    (ot.partial.partial_wasserstein / entropic_partial_wasserstein)

References:
  [1] Nguyen et al., "On Transportation of Mini-batches: A Hierarchical Approach", ICML 2022.
  [2] Nguyen et al., "Improving Mini-batch Optimal Transport via Partial Transportation", ICML 2022.
"""

import os

import numpy as np
import ot                       # POT: Python Optimal Transport library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from mirror_sinkhorn import mirror_sinkhorn
from tqdm import tqdm
from utils import model_eval, save_acc


class DigitsDA:
    """
    Domain Adaptation trainer for digit classification tasks (e.g. SVHN→MNIST).

    The training loss has two components:
      1. Classification loss:  CrossEntropy on labelled source data.
      2. Domain alignment loss: OT-based cost that aligns source and target
         feature embeddings (from model_g) and target pseudo-label predictions
         (from model_f).

    The total ground cost for OT between a source mini-batch and a target
    mini-batch is:
        C(i,j) = η₁ · ‖g(xˢᵢ) − g(xᵗⱼ)‖²  +  η₂ · (−yˢᵢ · log p(xᵗⱼ))
    where g is the feature extractor, p is the softmax prediction, and yˢ is
    the one-hot source label.
    """

    def __init__(
        self,
        model_g,
        model_f,
        n_class,
        logger,
        out_dir,
        eta1=0.1,
        eta2=0.1,
        epsilon=0.1,
        batch_epsilon=0.0,
        mass=0.5,
        tau=1.0,
        test_interval=10,
        use_mirror_sinkhorn=False,
        ms_iter=500,
    ):
        """
        Args:
            model_g:        Feature extractor (CNN backbone) mapping images → 128-dim embeddings.
            model_f:        Classifier head mapping 128-dim embeddings → n_class logits.
            n_class:        Number of classes (e.g. 10 for digit datasets).
            logger:         Python logger for recording metrics.
            out_dir:        Directory to save checkpoints and accuracy CSV.
            eta1:           Weight for the embedding distance cost  ‖g(xs) − g(xt)‖².
            eta2:           Weight for the label transport cost  −y_s · log(softmax(f(g(xt)))).
            epsilon:        Sinkhorn regularization for the *inner* mini-batch OT.
                            Set to 0 for exact OT (Earth Mover's Distance).
            batch_epsilon:  Sinkhorn regularization for the *outer* k×k OT in BoMb.
                            Set to 0 for exact matching between mini-batches.
            mass:           Fraction of mass to transport (only used by Partial OT, "jpmbot").
            tau:            Marginal relaxation penalty (only used by Unbalanced OT, "jumbot").
            test_interval:  Evaluate on the test set every this many epochs.
        """
        self.model_g = model_g      # Feature extractor (generator)
        self.model_f = model_f      # Classification head
        self.n_class = n_class
        self.logger = logger
        self.out_dir = out_dir
        self.out_file = os.path.join(self.out_dir, "acc.csv")
        # Remove stale accuracy file from previous runs
        if os.path.exists(self.out_file):
            os.remove(self.out_file)
        self.eta1 = eta1                # Embedding alignment weight
        self.eta2 = eta2                # Label alignment weight
        self.epsilon = epsilon          # Inner OT regularization
        self.batch_epsilon = batch_epsilon  # Outer (BoMb) OT regularization
        self.mass = mass                # Partial OT mass fraction
        self.tau = tau                  # Unbalanced OT marginal penalty
        self.test_interval = test_interval
        # Mirror Sinkhorn (Ballu & Berthet, ICML 2023) — replaces inner OT solver.
        self.use_mirror_sinkhorn = use_mirror_sinkhorn
        self.ms_iter = ms_iter
        self.logger.info("eta1, eta2, epsilon : {}, {}, {}".format(self.eta1, self.eta2, self.epsilon))
        self.logger.info(
            "use_mirror_sinkhorn = {}, ms_iter = {}".format(self.use_mirror_sinkhorn, self.ms_iter)
        )

    def _solve_inner_ot(self, a, b, C_np, method):
        """Mini-batch OT inner solver (shared by fit, fit_bomb, fit_bomb2).

        Dispatches to Mirror Sinkhorn (Ballu & Berthet, ICML 2023) when
        `self.use_mirror_sinkhorn` is True, else to the original POT
        solvers.  Single source of truth for the inner OT computation.
        """
        if self.use_mirror_sinkhorn:
            return mirror_sinkhorn(
                method, a, b, C_np,
                mass=self.mass, tau=self.tau, n_iter=self.ms_iter,
            )
        if method == "jumbot":
            return ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, C_np, self.epsilon, self.tau)
        if method == "jdot":
            if self.epsilon == 0:
                return ot.emd(a, b, C_np)
            return ot.sinkhorn(a, b, C_np, reg=self.epsilon)
        if method == "jpmbot":
            if self.epsilon == 0:
                return ot.partial.partial_wasserstein(a, b, C_np, self.mass)
            _M = C_np / (C_np.max() + 1e-10)
            return ot.partial.entropic_partial_wasserstein(
                a, b, _M, m=self.mass, reg=self.epsilon
            )
        raise ValueError(f"Unknown method: {method}")

    def fit(
        self,
        source_loader,
        target_loader,
        test_loader,
        n_epochs,
        criterion=nn.CrossEntropyLoss(),
        lr=2e-4,
        k=1,
        batch_size=25,
        method="jumbot",
    ):
        """
        Standard mini-batch OT (mOT) training — the "averaging" scheme.

        The full batch of size (k × m) is split into k mini-batches of size m.
        For each mini-batch index i, the i-th source mini-batch is paired with
        the i-th target mini-batch (diagonal pairing only). The OT loss for each
        pair is computed independently and averaged (scaled by 1/k).

        Total loss per iteration:
            L = (1/k) Σᵢ [ CrossEntropy(f(g(xˢᵢ)), yˢᵢ)  +  ⟨πᵢ, Cᵢ⟩ ]

        where πᵢ is the OT plan between the i-th source and target mini-batches
        and Cᵢ is the ground cost matrix.

        Args:
            source_loader:  DataLoader for labelled source domain.
            target_loader:  DataLoader for unlabelled target domain.
            test_loader:    DataLoader for target test set (evaluation only).
            n_epochs:       Number of training epochs.
            criterion:      Classification loss (default: CrossEntropyLoss).
            lr:             Learning rate for Adam optimizer.
            k:              Number of mini-batches to split each full batch into.
            batch_size:     Total batch size (= k × mini-batch size m).
            method:         Inner OT solver: "jdot" | "jumbot" | "jpmbot".
        """
        criterion = nn.CrossEntropyLoss()
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)
        best_acc = 0

        for id_epoch in range(n_epochs):
            print(f"Epoch: {id_epoch}")
            self.model_g.train()
            self.model_f.train()
            # Create a fresh iterator over the target loader each epoch
            target_loader_iter = iter(target_loader)

            for i, data in tqdm(enumerate(source_loader)):
                # ── Load source and target batches ────────────────────────
                xs_mb_all, ys_all = data  # source images + labels (size: batch_size)
                try:
                    xt_mb_all, _ = next(target_loader_iter)  # target images (labels unused)
                except StopIteration:
                    xt_mb_all = None
                if xt_mb_all is None or len(xt_mb_all) != batch_size:
                    # Target dataset exhausted or last batch is incomplete → restart
                    target_loader_iter = iter(target_loader)
                    xt_mb_all, _ = next(target_loader_iter)

                # Split the full batch indices [0, ..., batch_size-1] into k equal chunks.
                # inds_xs[i] gives the indices for the i-th mini-batch.
                inds_xs = np.split(np.array(range(xs_mb_all.shape[0])), k)

                # ── Forward pass + gradient accumulation across k mini-batches ──
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                for i in range(k):
                    total_loss = 0

                    # --- Source mini-batch: extract features and predict ---
                    xs_mb = xs_mb_all[inds_xs[i]].cuda()       # Source images  (m, C, H, W)
                    g_xs_mb = self.model_g(xs_mb)               # Source features (m, 128)
                    f_g_xs_mb = self.model_f(g_xs_mb)           # Source logits   (m, n_class)
                    ys = ys_all[inds_xs[i]].cuda()              # Source labels   (m,)

                    # Supervised classification loss, averaged over k mini-batches
                    s_loss = 1.0 / k * criterion(f_g_xs_mb, ys)
                    total_loss += s_loss

                    # --- Target mini-batch: extract features and predict ---
                    # Note: mOT uses diagonal pairing — same index i for source and target
                    xt_mb = xt_mb_all[inds_xs[i]].cuda()       # Target images  (m, C, H, W)
                    g_xt_mb = self.model_g(xt_mb)               # Target features (m, 128)
                    f_g_xt_mb = self.model_f(g_xt_mb)           # Target logits   (m, n_class)
                    pred_xt = F.softmax(f_g_xt_mb, 1)           # Target probabilities (m, n_class)

                    # --- Compute the ground cost matrix C(i,j) ---
                    # Component 1: squared Euclidean distance between feature embeddings
                    #   embed_cost[i,j] = ‖g(xˢᵢ) − g(xᵗⱼ)‖²
                    embed_cost = torch.cdist(g_xs_mb, g_xt_mb) ** 2

                    # Component 2: label-prediction cross-entropy cost
                    #   t_cost[i,j] = −yˢᵢ · log p(xᵗⱼ)
                    # where yˢ is one-hot and p is softmax prediction on target
                    ys_oh = F.one_hot(ys, num_classes=self.n_class).float()
                    t_cost = -torch.mm(ys_oh, torch.transpose(torch.log(pred_xt), 0, 1))

                    # Weighted combination of embedding and label costs
                    total_cost = self.eta1 * embed_cost + self.eta2 * t_cost

                    # --- Solve the inner OT problem ---
                    # Uniform marginals (each sample has equal mass 1/m)
                    a, b = ot.unif(g_xs_mb.size()[0]), ot.unif(g_xt_mb.size()[0])

                    pi = self._solve_inner_ot(
                        a, b, total_cost.detach().cpu().numpy(), method
                    )

                    # The OT plan π is computed without gradients (detached cost matrix),
                    # but the DA loss ⟨π, C⟩ IS differentiable w.r.t. the cost C
                    # because π is treated as a fixed weight matrix.
                    pi = torch.from_numpy(pi).float().cuda()
                    da_loss = 1.0 / k * torch.sum(pi * total_cost)  # Averaged over k
                    mloss = da_loss
                    total_loss += mloss

                    # Accumulate gradients from each mini-batch (backward without step)
                    total_loss.backward()

                # Single optimizer step after accumulating gradients from all k mini-batches
                optimizer_g.step()
                optimizer_f.step()

            # ── Periodic evaluation ───────────────────────────────────────
            if id_epoch % self.test_interval == 0 or (id_epoch == n_epochs - 1):
                source_acc = self.evaluate(source_loader)
                target_acc = self.evaluate(test_loader)
                self.logger.info(
                    "At epoch {} source and test accuracies are {} and {}".format(id_epoch, source_acc, target_acc)
                )
                save_acc(self.out_file, id_epoch, target_acc)
                # Save best model checkpoint based on target accuracy
                if target_acc > best_acc:
                    best_acc = target_acc
                    checkpoint = {
                        "model_g": self.model_g.state_dict(),
                        "model_f": self.model_f.state_dict(),
                        "epoch": id_epoch,
                        "accuracy": target_acc,
                    }
                    torch.save(checkpoint, os.path.join(self.out_dir, "best_model.pth"))

        # Save final checkpoint regardless of accuracy
        checkpoint = {
            "model_g": self.model_g.state_dict(),
            "model_f": self.model_f.state_dict(),
            "epoch": n_epochs,
            "accuracy": target_acc,
        }
        torch.save(checkpoint, os.path.join(self.out_dir, "final_model.pth"))

    def fit_bomb(
        self,
        source_loader,
        target_loader,
        test_loader,
        n_epochs,
        criterion=nn.CrossEntropyLoss(),
        lr=2e-4,
        k=1,
        batch_size=25,
        method="jumbot",
    ):
        """
        BoMb-OT training — Batch of Mini-batches with full outer plan weighting.

        Two-phase per iteration:
          Phase 1 (no grad): Compute all k² inner OT costs between every pair
              of source/target mini-batches. Build the k×k cost matrix and solve
              the outer OT to get the matching plan Π.
          Phase 2 (with grad): Re-forward all k² pairs. Weight each pair's DA
              loss by Π[i,j] and accumulate gradients.

        Total loss:
            L = Σᵢ Σⱼ [ (1/k²)·CE(f(g(xˢᵢ)), yˢᵢ)  +  Π[i,j]·⟨πᵢⱼ, Cᵢⱼ⟩ ]

        This is more expensive than mOT (O(k²) inner OT solves + two forward
        passes), but produces a better approximation of the full-batch OT.

        Args:
            source_loader:  DataLoader for labelled source domain.
            target_loader:  DataLoader for unlabelled target domain.
            test_loader:    DataLoader for target test set (evaluation only).
            n_epochs:       Number of training epochs.
            criterion:      Classification loss (default: CrossEntropyLoss).
            lr:             Learning rate for Adam optimizer.
            k:              Number of mini-batches (controls the k×k outer OT size).
            batch_size:     Total batch size (= k × mini-batch size m).
            method:         Inner OT solver: "jdot" | "jumbot" | "jpmbot".
        """
        criterion = nn.CrossEntropyLoss()
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)
        best_acc = 0

        for id_epoch in range(n_epochs):
            print(f"Epoch: {id_epoch}")
            self.model_g.train()
            self.model_f.train()
            target_loader_iter = iter(target_loader)

            for i, data in tqdm(enumerate(source_loader)):
                # ── Load source and target batches ────────────────────────
                xs_mb_all, ys_all = data
                try:
                    xt_mb_all, _ = next(target_loader_iter)
                except StopIteration:
                    xt_mb_all = None
                if xt_mb_all is None or len(xt_mb_all) != batch_size:
                    target_loader_iter = iter(target_loader)
                    xt_mb_all, _ = next(target_loader_iter)

                # Split both source and target into k mini-batches
                inds_xs = np.split(np.array(range(xs_mb_all.shape[0])), k)
                inds_xt = np.split(np.array(range(xt_mb_all.shape[0])), k)

                # ════════════════════════════════════════════════════════════
                # PHASE 1: Compute k×k cost matrix (no gradients needed)
                # ════════════════════════════════════════════════════════════
                list_da_loss = []
                with torch.no_grad():
                    for i in range(k):
                        # Source mini-batch i
                        xs_mb = xs_mb_all[inds_xs[i]].cuda()
                        g_xs_mb = self.model_g(xs_mb)
                        ys = ys_all[inds_xs[i]].cuda()

                        for j in range(k):
                            # Target mini-batch j
                            xt_mb = xt_mb_all[inds_xt[j]].cuda()
                            g_xt_mb = self.model_g(xt_mb)
                            f_g_xt_mb = self.model_f(g_xt_mb)
                            pred_xt = F.softmax(f_g_xt_mb, 1)

                            # Ground cost: η₁·‖embeddings‖² + η₂·(-label·log(pred))
                            embed_cost = torch.cdist(g_xs_mb, g_xt_mb) ** 2
                            ys_oh = F.one_hot(ys, num_classes=self.n_class).float()
                            t_cost = -torch.mm(ys_oh, torch.transpose(torch.log(pred_xt), 0, 1))
                            total_cost = self.eta1 * embed_cost + self.eta2 * t_cost

                            # Solve inner OT for this (i,j) mini-batch pair
                            a, b = ot.unif(g_xs_mb.size()[0]), ot.unif(g_xt_mb.size()[0])
                            pi = self._solve_inner_ot(
                                a, b, total_cost.detach().cpu().numpy(), method
                            )
                            pi = torch.from_numpy(pi).float().cuda()

                            # Inner OT cost for pair (i, j): scalar ⟨π, C⟩
                            da_loss = torch.sum(pi * total_cost)
                            list_da_loss.append(da_loss)

                    # ── Solve the OUTER k×k OT problem ────────────────────
                    # big_C[i,j] = OT cost between source mini-batch i and target mini-batch j
                    big_C = torch.stack(list_da_loss).view(k, k)
                    if self.batch_epsilon == 0:
                        # Exact OT between mini-batches (uniform marginals: each = 1/k)
                        plan = ot.emd([], [], big_C.detach().cpu().numpy())
                    else:
                        # Entropy-regularized OT between mini-batches
                        plan = ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=self.batch_epsilon)
                    # plan[i,j] = weight for pair (source_i, target_j) in the outer matching

                # ════════════════════════════════════════════════════════════
                # PHASE 2: Re-forward with gradients, weighted by outer plan
                # ════════════════════════════════════════════════════════════
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                for i in range(k):
                    for j in range(k):
                        total_loss = 0

                        # Source mini-batch i: classification loss
                        xs_mb = xs_mb_all[inds_xs[i]].cuda()
                        g_xs_mb = self.model_g(xs_mb)
                        f_g_xs_mb = self.model_f(g_xs_mb)
                        ys = ys_all[inds_xs[i]].cuda()
                        # Classification loss spread across all k² pairs
                        s_loss = 1.0 / (k**2) * criterion(f_g_xs_mb, ys)
                        total_loss += s_loss

                        # Skip DA loss if the outer plan assigns zero weight to (i,j)
                        if plan[i, j] == 0:
                            total_loss.backward()
                            continue

                        # Target mini-batch j: features + predictions
                        xt_mb = xt_mb_all[inds_xt[j]].cuda()
                        g_xt_mb = self.model_g(xt_mb)
                        f_g_xt_mb = self.model_f(g_xt_mb)
                        pred_xt = F.softmax(f_g_xt_mb, 1)

                        # Recompute ground cost (now on the computation graph for backprop)
                        embed_cost = torch.cdist(g_xs_mb, g_xt_mb) ** 2
                        ys_oh = F.one_hot(ys, num_classes=self.n_class).float()
                        t_cost = -torch.mm(ys_oh, torch.transpose(torch.log(pred_xt), 0, 1))
                        total_cost = self.eta1 * embed_cost + self.eta2 * t_cost

                        # Re-solve inner OT (plan is detached — treated as fixed weights)
                        a, b = ot.unif(g_xs_mb.size()[0]), ot.unif(g_xt_mb.size()[0])
                        pi = self._solve_inner_ot(
                            a, b, total_cost.detach().cpu().numpy(), method
                        )
                        pi = torch.from_numpy(pi).float().cuda()

                        # DA loss weighted by the outer plan: Π[i,j] · ⟨πᵢⱼ, Cᵢⱼ⟩
                        da_loss = torch.sum(pi * total_cost)
                        mloss = plan[i, j] * da_loss
                        total_loss += mloss
                        total_loss.backward()

                # Single step after accumulating all k² pair gradients
                optimizer_g.step()
                optimizer_f.step()

            # ── Periodic evaluation ───────────────────────────────────────
            if id_epoch % self.test_interval == 0 or (id_epoch == n_epochs - 1):
                source_acc = self.evaluate(source_loader)
                target_acc = self.evaluate(test_loader)
                self.logger.info(
                    "At epoch {} source and test accuracies are {} and {}".format(id_epoch, source_acc, target_acc)
                )
                save_acc(self.out_file, id_epoch, target_acc)
                if target_acc > best_acc:
                    best_acc = target_acc
                    checkpoint = {
                        "model_g": self.model_g.state_dict(),
                        "model_f": self.model_f.state_dict(),
                        "epoch": id_epoch,
                        "accuracy": target_acc,
                    }
                    torch.save(checkpoint, os.path.join(self.out_dir, "best_model.pth"))

        # Save final checkpoint regardless of accuracy
        checkpoint = {
            "model_g": self.model_g.state_dict(),
            "model_f": self.model_f.state_dict(),
            "epoch": n_epochs,
            "accuracy": target_acc,
        }
        torch.save(checkpoint, os.path.join(self.out_dir, "final_model.pth"))

    def fit_bomb2(
        self,
        source_loader,
        target_loader,
        test_loader,
        n_epochs,
        criterion=nn.CrossEntropyLoss(),
        lr=2e-4,
        k=1,
        batch_size=25,
        method="jumbot",
    ):
        """
        Stable BoMb-OT variant with deterministic 1-to-1 matching.

        Same Phase 1 as fit_bomb(): compute the k×k outer cost matrix and solve
        OT to get the outer plan Π. But instead of weighting all k² pairs,
        this method uses argmax on each row of Π to get a *permutation*
        mapping[i] = argmax_j Π[i,j], yielding a 1-to-1 source→target
        mini-batch assignment.

        Phase 2 then re-forwards only k matched pairs (i, mapping[i]) instead
        of all k² pairs, which:
          - Reduces computation from O(k²) to O(k) forward passes.
          - Avoids multiplying small plan weights with small losses (numerical
            stability when losses are near zero).

        Trade-off: Does not work well with the entropic (regularized) outer OT
        because Sinkhorn plans are dense — argmax may lose important probability
        mass spread across multiple columns.

        Total loss:
            L = (1/k) Σᵢ [ CE(f(g(xˢᵢ)), yˢᵢ)  +  Π[i,mapping[i]]·⟨πᵢ, Cᵢ⟩ ]

        Args:
            source_loader:  DataLoader for labelled source domain.
            target_loader:  DataLoader for unlabelled target domain.
            test_loader:    DataLoader for target test set (evaluation only).
            n_epochs:       Number of training epochs.
            criterion:      Classification loss (default: CrossEntropyLoss).
            lr:             Learning rate for Adam optimizer.
            k:              Number of mini-batches.
            batch_size:     Total batch size (= k × mini-batch size m).
            method:         Inner OT solver: "jdot" | "jumbot" | "jpmbot".
        """
        criterion = nn.CrossEntropyLoss()
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)
        best_acc = 0

        for id_epoch in range(n_epochs):
            print(f"Epoch: {id_epoch}")
            self.model_g.train()
            self.model_f.train()
            target_loader_iter = iter(target_loader)

            for i, data in tqdm(enumerate(source_loader)):
                # ── Load source and target batches ────────────────────────
                xs_mb_all, ys_all = data
                try:
                    xt_mb_all, _ = next(target_loader_iter)
                except StopIteration:
                    xt_mb_all = None
                if xt_mb_all is None or len(xt_mb_all) != batch_size:
                    target_loader_iter = iter(target_loader)
                    xt_mb_all, _ = next(target_loader_iter)

                # Split both source and target into k mini-batches
                inds_xs = np.split(np.array(range(xs_mb_all.shape[0])), k)
                inds_xt = np.split(np.array(range(xt_mb_all.shape[0])), k)

                # ════════════════════════════════════════════════════════════
                # PHASE 1: Build k×k cost matrix + solve outer OT (no grad)
                # ════════════════════════════════════════════════════════════
                list_da_loss = []
                with torch.no_grad():
                    for i in range(k):
                        # Source mini-batch i
                        xs_mb = xs_mb_all[inds_xs[i]].cuda()
                        g_xs_mb = self.model_g(xs_mb)
                        ys = ys_all[inds_xs[i]].cuda()

                        for j in range(k):
                            # Target mini-batch j
                            xt_mb = xt_mb_all[inds_xt[j]].cuda()
                            g_xt_mb = self.model_g(xt_mb)
                            f_g_xt_mb = self.model_f(g_xt_mb)
                            pred_xt = F.softmax(f_g_xt_mb, 1)

                            # Ground cost: η₁·‖embeddings‖² + η₂·(-label·log(pred))
                            embed_cost = torch.cdist(g_xs_mb, g_xt_mb) ** 2
                            ys_oh = F.one_hot(ys, num_classes=self.n_class).float()
                            t_cost = -torch.mm(ys_oh, torch.transpose(torch.log(pred_xt), 0, 1))
                            total_cost = self.eta1 * embed_cost + self.eta2 * t_cost

                            # Solve inner OT for pair (i, j)
                            a, b = ot.unif(g_xs_mb.size()[0]), ot.unif(g_xt_mb.size()[0])
                            pi = self._solve_inner_ot(
                                a, b, total_cost.detach().cpu().numpy(), method
                            )
                            pi = torch.from_numpy(pi).float().cuda()
                            da_loss = torch.sum(pi * total_cost)
                            list_da_loss.append(da_loss)

                    # ── Solve the OUTER k×k OT problem ────────────────────
                    big_C = torch.stack(list_da_loss).view(k, k)
                    if self.batch_epsilon == 0:
                        plan = ot.emd([], [], big_C.detach().cpu().numpy())
                    else:
                        plan = ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=self.batch_epsilon)

                    # KEY DIFFERENCE from fit_bomb():
                    # Extract a deterministic 1-to-1 mapping via argmax.
                    # mapping[i] = index of the target mini-batch assigned to source mini-batch i.
                    # For exact OT (batch_epsilon=0), the plan is a permutation matrix,
                    # so argmax perfectly recovers the matching.
                    mapping = np.argmax(plan, axis=1)

                # ════════════════════════════════════════════════════════════
                # PHASE 2: Re-forward only the k matched pairs (with grad)
                # ════════════════════════════════════════════════════════════
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                for i in range(k):
                    j = mapping[i]  # Target mini-batch matched to source mini-batch i
                    total_loss = 0

                    # Source mini-batch i: classification loss
                    xs_mb = xs_mb_all[inds_xs[i]].cuda()
                    g_xs_mb = self.model_g(xs_mb)
                    f_g_xs_mb = self.model_f(g_xs_mb)
                    ys = ys_all[inds_xs[i]].cuda()
                    # Classification loss averaged over k mini-batches
                    s_loss = 1.0 / k * criterion(f_g_xs_mb, ys)
                    total_loss += s_loss

                    # Target mini-batch mapping[i]: features + predictions
                    xt_mb = xt_mb_all[inds_xt[j]].cuda()
                    g_xt_mb = self.model_g(xt_mb)
                    f_g_xt_mb = self.model_f(g_xt_mb)
                    pred_xt = F.softmax(f_g_xt_mb, 1)

                    # Recompute ground cost (on the computation graph)
                    embed_cost = torch.cdist(g_xs_mb, g_xt_mb) ** 2
                    ys_oh = F.one_hot(ys, num_classes=self.n_class).float()
                    t_cost = -torch.mm(ys_oh, torch.transpose(torch.log(pred_xt), 0, 1))
                    total_cost = self.eta1 * embed_cost + self.eta2 * t_cost

                    # Re-solve inner OT (π is detached, used as fixed weights)
                    a, b = ot.unif(g_xs_mb.size()[0]), ot.unif(g_xt_mb.size()[0])
                    pi = self._solve_inner_ot(
                        a, b, total_cost.detach().cpu().numpy(), method
                    )
                    pi = torch.from_numpy(pi).float().cuda()

                    # DA loss weighted by the outer plan entry for this matched pair
                    da_loss = torch.sum(pi * total_cost)
                    mloss = plan[i, j] * da_loss
                    total_loss += mloss
                    total_loss.backward()

                # Single optimizer step after all k matched-pair gradients
                optimizer_g.step()
                optimizer_f.step()

            # ── Periodic evaluation ───────────────────────────────────────
            if id_epoch % self.test_interval == 0 or (id_epoch == n_epochs - 1):
                source_acc = self.evaluate(source_loader)
                target_acc = self.evaluate(test_loader)
                self.logger.info(
                    "At epoch {} source and test accuracies are {} and {}".format(id_epoch, source_acc, target_acc)
                )
                save_acc(self.out_file, id_epoch, target_acc)
                if target_acc > best_acc:
                    best_acc = target_acc
                    checkpoint = {
                        "model_g": self.model_g.state_dict(),
                        "model_f": self.model_f.state_dict(),
                        "epoch": id_epoch,
                        "accuracy": target_acc,
                    }
                    torch.save(checkpoint, os.path.join(self.out_dir, "best_model.pth"))

        # Save final checkpoint regardless of accuracy
        checkpoint = {
            "model_g": self.model_g.state_dict(),
            "model_f": self.model_f.state_dict(),
            "epoch": n_epochs,
            "accuracy": target_acc,
        }
        torch.save(checkpoint, os.path.join(self.out_dir, "final_model.pth"))

    def source_only(self, source_loader, criterion=nn.CrossEntropyLoss(), lr=2e-4):
        """
        Pre-train on labelled source data only (no domain adaptation).

        Runs 10 epochs of supervised cross-entropy training to give the feature
        extractor and classifier a reasonable initialization before the OT-based
        domain adaptation phase begins. This warm-start helps stabilize the
        initial OT cost computation.

        Args:
            source_loader:  DataLoader for labelled source domain.
            criterion:      Loss function (default: CrossEntropyLoss).
            lr:             Learning rate.
        """
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)

        for _ in tqdm(range(10)):
            self.model_g.train()
            self.model_f.train()
            for _, data in enumerate(source_loader):
                # Load source batch
                xs_mb, ys = data
                xs_mb, ys = xs_mb.cuda(), ys.cuda()

                # Forward: image → features → logits
                g_xs_mb = self.model_g(xs_mb)
                f_g_xs_mb = self.model_f(g_xs_mb)

                # Standard supervised classification loss
                s_loss = criterion(f_g_xs_mb, ys)

                # Backward + update
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                tot_loss = s_loss
                tot_loss.backward()

                optimizer_g.step()
                optimizer_f.step()

        source_acc = self.evaluate(source_loader)
        self.logger.info("Source accuracy is {}".format(source_acc))

    def evaluate(self, data_loader):
        """Evaluate classification accuracy on a given dataset."""
        score = model_eval(data_loader, self.model_g, self.model_f)
        return score
