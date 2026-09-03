from __future__ import print_function

import argparse
import logging
import os
import random

import numpy as np
import torch
import torchvision.datasets as datasets
from Mnist_generator import MnistGenerator
from torch import optim
from torchvision import transforms
from tqdm import tqdm
from utils import compute_true_Wasserstein, save_acc


def main():
    parser = argparse.ArgumentParser(description="MNIST generative model")
    parser.add_argument("--datadir", default=".data")
    parser.add_argument("--outdir", default="./result")
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--latent-size", type=int, default=128)
    parser.add_argument("--fid-each", type=int, default=5)
    parser.add_argument("--L", type=int, default=1000)
    parser.add_argument("--method", type=str, default="OT")
    parser.add_argument("--bomb", action="store_true")
    parser.add_argument("--reg", type=float, default=1)
    parser.add_argument("--ebomb", action="store_true")
    parser.add_argument("--breg", type=float, default=1)
    parser.add_argument("--tau", type=float, default=1)
    parser.add_argument("--mass", type=float, default=0.65)
    # KPG-RL parameters
    parser.add_argument("--use-kpg", action="store_true")
    parser.add_argument("--n-kp", type=int, default=5, help="number of keypoint pairs")
    parser.add_argument("--alpha", type=float, default=0.5, help="KPG blending coefficient")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    np.random.seed(args.seed); random.seed(args.seed); torch.random.manual_seed(args.seed)
    method = args.method
    latent_size = args.latent_size
    args.epochs = args.epochs * args.k

    kpg_tag = f"_kpg_nkp{args.n_kp}_a{args.alpha}" if args.use_kpg else ""
    description = (
        f"Mnist_{method}_k{args.k}_m{args.m}_reg{args.reg}_tau{args.tau}"
        f"_mass{args.mass}_L{args.L}_seed{args.seed}_{args.epochs}epochs{kpg_tag}"
    )
    bomb, ebomb = False, False
    if args.bomb:
        bomb = True; description = "BoMb-" + description
    elif args.ebomb:
        ebomb = True; description = f"eBoMb{args.breg}-" + description

    model_dir = os.path.join(args.outdir, description)
    LOG_DIR = "logs/mnist"; CSV_DIR = "csv/mnist"
    for d in [LOG_DIR, CSV_DIR, args.datadir, args.outdir, model_dir]:
        os.makedirs(d, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{description}.log")
    csv_file = os.path.join(CSV_DIR, f"{description}.csv")
    for f in [log_file, csv_file]:
        if os.path.exists(f): os.remove(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(filename=log_file, filemode="a",
                        format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p",
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.info(f"Parameters: {args}")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.datadir, train=True, download=True,
                        transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.k * args.m, shuffle=True, num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.datadir, train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=10000, shuffle=True, num_workers=args.num_workers,
    )

    model = MnistGenerator(
        image_size=28, latent_size=latent_size, hidden_size=100, device=device,
        use_kpg=args.use_kpg, n_kp=args.n_kp, alpha=args.alpha,
        kpg_seed=args.seed,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        total_g_loss = 0.0
        logger.info(f"Epoch: {epoch}"); print(f"Epoch: {epoch}")
        for batch_idx, (data, y) in tqdm(enumerate(train_loader, start=0)):
            g_loss = model.train_minibatch(
                optimizer, data, args.k, args.m, method,
                args.reg, args.breg, args.tau, args.mass, args.L, bomb, ebomb,
            )
            total_g_loss += g_loss.item()
        total_g_loss /= batch_idx + 1
        logger.info(f"{method} Epoch: {epoch}, G Loss: {total_g_loss}")

        if (epoch % args.fid_each == 0) or (epoch == args.epochs - 1):
            save_m_dir = model_dir + "/models"
            os.makedirs(save_m_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_m_dir}/G_{epoch:06d}.pth")
            model.eval()
            for _, (input, y) in enumerate(test_loader, start=0):
                fixednoise_wd = torch.randn((10000, latent_size)).to(device)
                data = input.to(device).view(input.shape[0], -1)
                fake = model.decoder(fixednoise_wd)
                W = compute_true_Wasserstein(data.cpu(), fake.view(data.shape[0], -1).cpu())
                break
            model.train()
            logger.info(f"Wasserstein score: {W}")
            save_acc(csv_file, epoch, W)


if __name__ == "__main__":
    main()
