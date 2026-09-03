"""CIFAR-10 generator + discriminator with optional KPG-RL guided OT.

Architecture is unchanged from the baseline.  The only modification is that
every inline OT solver call is routed through _solve(), which dispatches to
the KPG-RL two-pass solver when use_kpg=True.  KPG relation profiles are
computed in discriminator feature space.
"""

import numpy as np
import ot
import torch
import torch.nn as nn
from kpg_ot import (solve_ot, solve_kuot_paper,
                    _select_keypoint_pairs_from_plan, check_batch_feasibility)
from utils import sliced_wasserstein_distance


class Discriminator(nn.Module):
    def __init__(self, image_size, latent_size, num_chanel, hidden_chanels=64):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.latent_size = latent_size
        self.num_chanel = num_chanel
        self.hidden_chanels = hidden_chanels
        self.main1 = nn.Sequential(
            nn.Conv2d(self.num_chanel, self.hidden_chanels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hidden_chanels, self.hidden_chanels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hidden_chanels * 2, self.hidden_chanels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 4),
            nn.Tanh(),
        )
        self.main2 = nn.Sequential(
            nn.Conv2d(self.hidden_chanels * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.main1(x)
        y = self.main2(h).view(x.shape[0], -1)
        return y, h


class Generator(nn.Module):
    def __init__(self, image_size, latent_size, num_chanel, hidden_chanels=64):
        super(Generator, self).__init__()
        self.image_size = image_size
        self.latent_size = latent_size
        self.num_chanel = num_chanel
        self.hidden_chanels = hidden_chanels
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, self.hidden_chanels * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.hidden_chanels * 4, self.hidden_chanels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.hidden_chanels * 2, self.hidden_chanels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.hidden_chanels, self.num_chanel, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z.view(z.shape[0], self.latent_size, 1, 1))


class Cifar_Generator(nn.Module):
    def __init__(self, image_size, latent_size, num_chanel, hidden_chanels, device,
                 use_kpg=False, n_kp=5, alpha=0.5, kpg_seed=16,
                 rho=0.1, kp_metric="euclidean",
                 kuot_iters=1000):
        super(Cifar_Generator, self).__init__()
        self.image_size = image_size
        self.num_chanel = num_chanel
        self.latent_size = latent_size
        self.hidden_chanels = hidden_chanels
        self.device = device
        self.decoder = Generator(image_size, latent_size, num_chanel, hidden_chanels)
        self.use_kpg = use_kpg
        self.n_kp = n_kp
        self.alpha = alpha
        self.kpg_rng   = np.random.default_rng(kpg_seed)
        # Sec. IV formulation state
        self.rho         = rho
        self.kp_metric   = kp_metric
        self.kuot_iters  = kuot_iters
        self.kp_real     = None      # k fixed real images
        self.kp_z        = None      # k fixed latent codes

    # ----- Sec. IV: fixed keypoint pairs (real image, latent code) -------

    def _init_keypoints(self, data_mb, z_mb, discriminator, method, reg, tau, mass):
        """Mine the k FIXED keypoint pairs once, from an initial transport plan.

        DeepGM has no labels, so Sec. IV-A's annotated pairs are obtained the way
        Sec. V-E describes: solve the unguided problem and keep its highest-mass
        correspondences.  We then store the real IMAGES and the latent CODES, so the pairs
        stay fixed for the whole run while the generated member's features move with the
        generator — the exact analogue of the DA keypoints moving with the encoder.
        """
        with torch.no_grad():
            fake_mb = self.decoder(z_mb)
            _, fd = discriminator(data_mb)
            _, ff = discriminator(fake_mb)
            fd = fd.view(data_mb.size(0), -1)
            ff = ff.view(z_mb.size(0), -1)
            cost = torch.cdist(fd, ff) ** 2
            a, b = ot.unif(cost.size(0)), ot.unif(cost.size(1))
            pi = solve_ot(a, b, cost.detach().cpu().numpy(), method, reg, tau, mass)
        I_kp, J_kp = _select_keypoint_pairs_from_plan(pi, self.n_kp)
        if len(I_kp) == 0:
            raise RuntimeError("keypoint mining found no usable pair in the initial plan")
        self.kp_real = data_mb[np.asarray(I_kp, dtype=int)].detach().clone()
        self.kp_z = z_mb[np.asarray(J_kp, dtype=int)].detach().clone()
        print("[mkuot] DeepGM: mined {} fixed keypoint pairs (real image, latent z)"
              .format(self.kp_real.size(0)))

    def prepare_keypoint_batches(self, data, z, inds_data, inds_z,
                                 discriminator, method, reg, tau, mass):
        """Write the fixed keypoints into the first k slots of EVERY sub-batch (Eq. (5)).

        Called once per `train_minibatch`, before any feature is computed, so all k x k
        sub-batch pairs see the keypoints aligned at positions 0..k-1.
        """
        if self.kp_real is None:
            self._init_keypoints(data[inds_data[0]].to(self.device),
                                 z[inds_z[0]].cuda(self.device),
                                 discriminator, method, reg, tau, mass)
        k_kp = self.kp_real.size(0)
        data, z = data.clone(), z.clone()
        for ids in inds_data:
            n = min(k_kp, len(ids))
            check_batch_feasibility(len(ids), n, strict=False)
            data[ids[:n]] = self.kp_real[:n].to(data.device, data.dtype)
        for ids in inds_z:
            n = min(k_kp, len(ids))
            z[ids[:n]] = self.kp_z[:n].to(z.device, z.dtype)
        return data, z

    def _solve(self, cost_matrix, method, reg, tau, mass, feat_real=None, feat_fake=None):
        a, b = ot.unif(cost_matrix.size(0)), ot.unif(cost_matrix.size(1))
        C_np = cost_matrix.detach().cpu().numpy()
        if self.use_kpg and feat_real is not None and feat_fake is not None:
            # Sec. IV: the batch already carries the k fixed keypoints in its first k
            # slots, so apply the mask (Eq. 6) + guiding matrix (Eq. 9) + blend (Eq. 10)
            # and solve on the masked support.  `--method OT` (the setting these
            # experiments use) is the BALANCED regime -> m-KOT.
            k_kp = 0 if self.kp_real is None else self.kp_real.size(0)
            return torch.from_numpy(solve_kuot_paper(
                C_np, feat_real, feat_fake, k_kp,
                alpha=self.alpha, rho=self.rho, metric=self.kp_metric,
                method=method, reg=reg, tau=tau, mass=mass,
                n_iter=self.kuot_iters,
            )).float().cuda(self.device)
        return torch.from_numpy(solve_ot(a, b, C_np, method, reg, tau, mass)).cuda(self.device)

    def _ot_block(self, feature_data, feature_fake, cost_matrix, method, reg, tau, mass):
        """Solve OT and return (pi, loss) as tensors."""
        feat_r = feature_data.detach().cpu().numpy()
        feat_f = feature_fake.detach().cpu().numpy()
        pi = self._solve(cost_matrix, method, reg, tau, mass, feat_r, feat_f)
        return pi, torch.sum(pi * cost_matrix)

    def train_minibatch(
        self, model_op, discriminator, optimizer, data, k, m,
        method="OT", reg=0, breg=0, tau=1, mass=0.65, L=1000,
        bomb=False, ebomb=False,
    ):
        z = torch.randn((data.shape[0], self.latent_size))
        if (data.shape[0] % k) == 0:
            inds_data = np.split(np.arange(data.shape[0]), k)
            inds_z = np.split(np.arange(z.shape[0]), k)
        else:
            real_k = int(data.shape[0] / m)
            if real_k != 0:
                inds_data = list(np.split(np.arange(real_k * m), real_k))
                inds_z = list(np.split(np.arange(real_k * m), real_k))
                k = real_k
                if method != "sliced" and data.shape[0] % m != 0:
                    inds_data.append(np.arange(real_k * m, data.shape[0]))
                    inds_z.append(np.arange(real_k * m, data.shape[0]))
                    k += 1
            else:
                k = 1
                inds_data = [np.arange(data.shape[0])]
                inds_z = [np.arange(data.shape[0])]

        # Sec. IV-A / Eq. (5): put the k FIXED keypoint pairs in the first k slots
        # of every sub-batch before any feature is computed.
        if self.use_kpg:
            data, z = self.prepare_keypoint_batches(
                data, z, inds_data, inds_z, discriminator, method, reg, tau, mass)

        # ── Discriminator phase ──────────────────────────────────────────
        dloss = []
        if (bomb or ebomb) and method != "sliced":
            self.eval(); discriminator.eval()
            with torch.no_grad():
                for i in range(k):
                    for j in range(k):
                        data_mb = data[inds_data[i]].to(self.device)
                        z_mb = z[inds_z[j]].cuda(self.device)
                        fake_mb = self.decoder(z_mb)
                        _, fd = discriminator(data_mb)
                        _, ff = discriminator(fake_mb)
                        fd = fd.view(data_mb.size(0), -1); ff = ff.view(z_mb.size(0), -1)
                        cost = torch.cdist(fd, ff) ** 2
                        pi, loss = self._ot_block(fd, ff, cost, method, reg, tau, mass)
                        dloss.append(loss)
                big_C = torch.stack(dloss).view(k, k)
                plan = ot.emd([], [], big_C.detach().cpu().numpy()) if bomb else \
                       ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=breg)

        Dloss = 0
        self.train(); discriminator.train()
        if method == "sliced":
            optimizer.zero_grad()
            for i in range(k):
                data_mb = data[inds_data[i]].to(self.device)
                y_data, _ = discriminator(data_mb)
                label = torch.full((data_mb.shape[0], 1), 1, dtype=torch.float32, device=self.device)
                (1.0 / (k**2) * nn.BCELoss(reduction="sum")(y_data, label)).backward()
            optimizer.step()
            optimizer.zero_grad()
            for j in range(k):
                z_mb = z[inds_z[j]].cuda(self.device)
                fake_mb = self.decoder(z_mb)
                y_fake, _ = discriminator(fake_mb)
                label = torch.full((z_mb.shape[0], 1), 0, dtype=torch.float32, device=self.device)
                (1.0 / (k**2) * nn.BCELoss(reduction="sum")(y_fake, label)).backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            for i in range(k):
                for j in range(k):
                    if (bomb or ebomb) and plan[i, j] == 0:
                        continue
                    data_mb = data[inds_data[i]].to(self.device)
                    z_mb = z[inds_z[j]].cuda(self.device)
                    fake_mb = self.decoder(z_mb)
                    _, fd = discriminator(data_mb); _, ff = discriminator(fake_mb)
                    fd = fd.view(data_mb.size(0), -1); ff = ff.view(z_mb.size(0), -1)
                    cost = torch.cdist(fd, ff) ** 2
                    pi, loss = self._ot_block(fd, ff, cost, method, reg, tau, mass)
                    w = plan[i, j] if (bomb or ebomb) else 1.0 / (k**2)
                    mloss = -w * loss
                    Dloss += mloss
                    mloss.backward()
            optimizer.step()

        # ── Generator phase ──────────────────────────────────────────────
        gloss = []
        if bomb or ebomb:
            with torch.no_grad():
                self.eval(); discriminator.eval()
                for i in range(k):
                    for j in range(k):
                        data_mb = data[inds_data[i]].to(self.device)
                        z_mb = z[inds_z[j]].cuda(self.device)
                        fake_mb = self.decoder(z_mb)
                        _, fd = discriminator(data_mb); _, ff = discriminator(fake_mb)
                        fd = fd.view(data_mb.size(0), -1); ff = ff.view(z_mb.size(0), -1)
                        if method == "sliced":
                            gloss.append(sliced_wasserstein_distance(fd, ff, num_projections=L, device=self.device))
                        else:
                            cost = torch.cdist(fd, ff) ** 2
                            _, loss = self._ot_block(fd, ff, cost, method, reg, tau, mass)
                            gloss.append(loss)
                big_C = torch.stack(gloss).view(k, k)
                plan = ot.emd([], [], big_C.detach().cpu().numpy()) if bomb else \
                       ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=breg)

        self.train(); discriminator.train()
        model_op.zero_grad()
        G_loss = 0
        for i in range(k):
            for j in range(k):
                if (bomb or ebomb) and plan[i, j] == 0:
                    continue
                data_mb = data[inds_data[i]].to(self.device)
                z_mb = z[inds_z[j]].cuda(self.device)
                fake_mb = self.decoder(z_mb)
                _, fd = discriminator(data_mb); _, ff = discriminator(fake_mb)
                fd = fd.view(data_mb.size(0), -1); ff = ff.view(z_mb.size(0), -1)
                if method == "sliced":
                    loss = sliced_wasserstein_distance(fd, ff, num_projections=L, device=self.device)
                else:
                    cost = torch.cdist(fd, ff) ** 2
                    _, loss = self._ot_block(fd, ff, cost, method, reg, tau, mass)
                w = plan[i, j] if (bomb or ebomb) else 1.0 / (k**2)
                mloss = w * loss
                G_loss += mloss
                mloss.backward()
        model_op.step()
        return G_loss, Dloss
