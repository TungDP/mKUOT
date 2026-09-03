"""MNIST generator — UNGUIDED mini-batch OT baseline (m-OT / m-UOT / m-POT).

Keypoint guidance is not available here: MNIST has no discriminator (OT runs in
flattened pixel space), so the plan-mined keypoint pairs of Sec. IV do not transfer, and
MNIST is not one of the paper's reported generative experiments.  Passing use_kpg=True
raises rather than running something the paper does not describe; see CelebA / CIFAR-10
for the guided setting.
"""

import numpy as np
import ot
import torch
import torch.nn as nn
from kpg_ot import solve_ot
from utils import sliced_wasserstein_distance


class Generator(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size):
        super(Generator, self).__init__()
        self.image_size = image_size**2
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(True),
            nn.Linear(2 * hidden_size, 4 * hidden_size),
            nn.ReLU(True),
            nn.Linear(4 * hidden_size, self.image_size),
            nn.ReLU(True),
        )

    def forward(self, input):
        return self.main(input)


class MnistGenerator(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size, device,
                 use_kpg=False, n_kp=5, alpha=0.5, kpg_seed=16):
        super(MnistGenerator, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.device = device
        self.decoder = Generator(image_size, hidden_size, latent_size)
        self.use_kpg = use_kpg
        self.n_kp = n_kp
        self.alpha = alpha
        self.kpg_rng   = np.random.default_rng(kpg_seed)
        # The Sec. IV formulation is NOT wired for MNIST: this model has no
        # discriminator (OT runs in flattened pixel space), so the keypoint mining used by
        # Celeba_Generator / Cifar_Generator does not transfer, and MNIST is not one of the
        # paper's reported generative experiments (Table 6 is CelebA + CIFAR-10).  Refuse
        # rather than run something the paper does not describe.
        if use_kpg:
            raise NotImplementedError(
                "MNIST does not support keypoint guidance.  This model has no "
                "discriminator (OT runs in flattened pixel space), so the plan-mined "
                "keypoint pairs used by Celeba_Generator / Cifar_Generator do not "
                "transfer, and MNIST is not one of the paper's reported generative "
                "experiments (CelebA and CIFAR-10 are).  Run it without --use-kpg as an "
                "unguided mini-batch OT baseline."
            )

    def _solve(self, cost_matrix, method, reg, tau, mass, feat_real=None, feat_fake=None):
        """Dispatch to standard or KPG-guided OT."""
        a, b = ot.unif(cost_matrix.size(0)), ot.unif(cost_matrix.size(1))
        C_np = cost_matrix.detach().cpu().numpy()

        pi = solve_ot(a, b, C_np, method, reg, tau, mass)
        return torch.from_numpy(pi).cuda(self.device)

    def train_minibatch(
        self, model_op, data, k, m, method="OT", reg=0, breg=0,
        tau=1, mass=0.65, L=1000, bomb=False, ebomb=False,
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
                if method != "sliced":
                    inds_data.append(np.arange(real_k * m, data.shape[0]))
                    inds_z.append(np.arange(real_k * m, data.shape[0]))
                    k = k + 1
            else:
                k = 1
                inds_data = [np.arange(data.shape[0])]
                inds_z = [np.arange(data.shape[0])]

        gloss = []
        if bomb or ebomb:
            with torch.no_grad():
                self.eval()
                for i in range(k):
                    for j in range(k):
                        data_mb = data[inds_data[i]].to(self.device)
                        z_mb = z[inds_z[j]].cuda(self.device)
                        fake_mb = self.decoder(z_mb)
                        if method == "sliced":
                            gloss.append(sliced_wasserstein_distance(
                                data_mb.view(data_mb.shape[0], -1),
                                fake_mb.view(fake_mb.shape[0], -1),
                                num_projections=L, device=self.device,
                            ))
                        else:
                            feat_r = data_mb.view(data_mb.shape[0], -1)
                            feat_f = fake_mb.view(fake_mb.shape[0], -1)
                            cost_matrix = torch.cdist(feat_r, feat_f) ** 2
                            pi = self._solve(cost_matrix, method, reg, tau, mass,
                                             feat_r.cpu().numpy(), feat_f.cpu().numpy())
                            gloss.append(torch.sum(pi * cost_matrix))
                big_C = torch.stack(gloss).view(k, k)
                if bomb:
                    plan = ot.emd([], [], big_C.detach().cpu().numpy())
                elif ebomb:
                    plan = ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=breg)

        self.train()
        model_op.zero_grad()
        G_loss = 0
        for i in range(k):
            for j in range(k):
                if bomb or ebomb:
                    if plan[i, j] == 0:
                        continue
                data_mb = data[inds_data[i]].to(self.device)
                z_mb = z[inds_z[j]].cuda(self.device)
                fake_mb = self.decoder(z_mb)
                if method == "sliced":
                    loss = sliced_wasserstein_distance(
                        data_mb.view(data_mb.shape[0], -1),
                        fake_mb.view(fake_mb.shape[0], -1),
                        num_projections=L, device=self.device,
                    )
                else:
                    feat_r = data_mb.view(data_mb.shape[0], -1)
                    feat_f = fake_mb.view(fake_mb.shape[0], -1)
                    cost_matrix = torch.cdist(feat_r, feat_f) ** 2
                    pi = self._solve(cost_matrix, method, reg, tau, mass,
                                     feat_r.detach().cpu().numpy(),
                                     feat_f.detach().cpu().numpy())
                    loss = torch.sum(pi * cost_matrix)
                if bomb or ebomb:
                    mloss = plan[i, j] * loss
                else:
                    mloss = 1.0 / (k ** 2) * loss
                G_loss += mloss
                mloss.backward()
        model_op.step()
        return G_loss
