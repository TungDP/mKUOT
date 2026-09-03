from __future__ import print_function

import argparse
import csv
import logging
import os
import random
import imageio
import numpy as np
import torch
from Celeba_generator import Celeba_Generator, Discriminator
from experiments import sampling
from fid_score import calculate_fid_given_paths
from PIL import Image
from torch import optim
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from utils import save_acc


class CelebADataset(Dataset):
    _SPLIT_MAP = {"train": 0, "valid": 1, "test": 2}

    def __init__(self, root, split="train", transform=None):
        self.img_dir = os.path.join(root, "img_align_celeba")
        self.transform = transform
        split_id = self._SPLIT_MAP[split]
        with open(os.path.join(root, "list_eval_partition.csv")) as f:
            reader = csv.DictReader(f)
            self.filenames = [row["image_id"] for row in reader if int(row["partition"]) == split_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.filenames[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def main():
    parser = argparse.ArgumentParser(description="CelebA generative model")
    parser.add_argument("--datadir", default="/home/doanpt/locnd/Mini-batch_Keypoint-Guided-Relative_OT/Mini-batch_KPG-RL_OT/data/celeba")
    parser.add_argument("--outdir", default="./results")
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--latent-size", type=int, default=128)
    parser.add_argument("--fid-each", type=int, default=5)
    parser.add_argument("--method", type=str, default="OT")
    parser.add_argument("--bomb", action="store_true")
    parser.add_argument("--reg", type=float, default=1)
    parser.add_argument("--ebomb", action="store_true")
    parser.add_argument("--breg", type=float, default=1)
    parser.add_argument("--tau", type=float, default=1)
    parser.add_argument("--mass", type=float, default=0.65)
    parser.add_argument("--L", type=int, default=1000)
    # KPG-RL parameters
    parser.add_argument("--use-kpg", action="store_true")
    parser.add_argument("--n-kp", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    # ---- Sec. IV formulation ----------------------------
    parser.add_argument(
        "--rho", type=float, default=0.1,
        help="dimensionless relation-profile temperature of Eq. (8); softmax scale "
             "rho*max(c).")
    parser.add_argument(
        "--kp-metric", type=str, default="euclidean",
        choices=["euclidean", "sqeuclidean"],
        help="ground metric for the relation profiles of Eq. (8)")
    parser.add_argument(
        "--kuot-iters", type=int, default=1000,
        help="scaling iterations N in Alg. 1 (only used by the unbalanced solver)")
    args = parser.parse_args()

    # --gpu-id was parsed but never applied, so every run landed on CUDA device 0
    # regardless of the flag (which silently serialised parallel sweeps onto one
    # card).  torch is already imported here, so CUDA_VISIBLE_DEVICES no longer
    # takes effect; select the device explicitly instead.
    if torch.cuda.is_available():
        torch.cuda.set_device(int(str(args.gpu_id).split(",")[0]))
        print(f"[mkuot] using CUDA device {torch.cuda.current_device()}")

    np.random.seed(args.seed); random.seed(args.seed); torch.random.manual_seed(args.seed)
    method = args.method
    latent_size = args.latent_size
    args.epochs = args.epochs * args.k

    kpg_tag = f"_kpg_nkp{args.n_kp}_a{args.alpha}" if args.use_kpg else ""
    description = (
        f"CelebA_{method}_k{args.k}_m{args.m}_reg{args.reg}_tau{args.tau}"
        f"_mass{args.mass}_{args.L}_seed{args.seed}_{args.epochs}epochs{kpg_tag}"
    )
    bomb, ebomb = False, False
    if args.bomb:
        bomb = True; description = "BoMb-" + description
    elif args.ebomb:
        ebomb = True; description = f"eBoMb{args.breg}-" + description

    model_dir = os.path.join(args.outdir, description)
    LOG_DIR = "logs/celeba"; CSV_DIR = "csv/celeba"
    for d in [LOG_DIR, CSV_DIR, args.outdir, model_dir]:
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
        CelebADataset(args.datadir, split="train", transform=transforms.Compose([
            transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])),
        batch_size=args.m * args.k, shuffle=True, num_workers=args.num_workers,
    )

    model = Celeba_Generator(
        image_size=64, latent_size=latent_size, num_chanel=3, hidden_chanels=64, device=device,
        use_kpg=args.use_kpg, n_kp=args.n_kp, alpha=args.alpha,
        kpg_seed=args.seed,
        rho=args.rho,
        kp_metric=args.kp_metric, kuot_iters=args.kuot_iters,
    ).to(device)
    dis = Discriminator(64, latent_size, 3, 64).to(device)
    disoptimizer = optim.Adam(dis.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    fixednoise = torch.randn((64, latent_size)).to(device)
    best_fid, best_epoch = float("inf"), -1     # FID: lower is better

    for epoch in range(args.epochs):
        total_g_loss = 0.0; total_d_loss = 0.0
        logger.info(f"Epoch: {epoch}"); print(f"Epoch: {epoch}")
        for batch_idx, (data, y) in tqdm(enumerate(train_loader, start=0)):
            g_loss, d_loss = model.train_minibatch(
                optimizer, dis, disoptimizer, data, args.k, args.m,
                method, args.reg, args.breg, args.tau, args.mass, args.L, bomb, ebomb,
            )
            total_g_loss += float(g_loss); total_d_loss += float(d_loss)
        total_g_loss /= batch_idx + 1; total_d_loss /= batch_idx + 1
        logger.info(f"{method} Epoch: {epoch}, G Loss: {total_g_loss}, D Loss: {total_d_loss}")

        if (epoch % args.fid_each == 0) or (epoch == args.epochs - 1):
            save_m_dir = model_dir + "/models"
            os.makedirs(save_m_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_m_dir}/G_{epoch:06d}.pth")
            model.eval()
            with torch.no_grad():
                sampling(f"{model_dir}/sample_epoch_{epoch}.png", fixednoise, model.decoder, 64, 64, 3)
                outdir_images = model_dir + "/images"
                os.makedirs(outdir_images, exist_ok=True)
                count = 0
                for _ in tqdm(range(11000 // args.m)):
                    z2 = torch.randn(args.m, latent_size).to(device)
                    fake = model.decoder(z2).cpu().detach().numpy().reshape(-1, 3, 64, 64)
                    fake = ((fake.transpose((0, 2, 3, 1)) / 2.0 + 0.5) * 255).astype(np.uint8)
                    for img in fake:
                        imageio.imwrite(f"{outdir_images}/img_{count:06d}.png", img); count += 1
            model.train()
            logger.info(f"wrote images to {outdir_images}")
            torch.cuda.empty_cache()
            fid_stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../Baselines/Mini-batch-OT/DeepGM/fid_stats_celeba_test.npz")
            fid = calculate_fid_given_paths([outdir_images, fid_stats_path], 4, device, 2048, args.num_workers)
            logger.info(f"FID score: {fid}")
            save_acc(csv_file, epoch, fid)
            if fid < best_fid:
                best_fid, best_epoch = fid, epoch

    # ---- best-FID report (lower is better) --------------------------------
    summary = f"BEST FID: {best_fid:.4f} @ epoch {best_epoch}  [{description}]"
    logger.info(summary)
    print(summary)
    # One line per completed run, appended to a per-dataset summary CSV so the
    # run scripts can tabulate "best FID per setting" without re-parsing logs.
    summary_csv = os.path.join(CSV_DIR, "best_fid_summary.csv")
    write_header = not os.path.exists(summary_csv)
    with open(summary_csv, "a") as f:
        if write_header:
            f.write("setting,best_epoch,best_fid\n")
        f.write(f"{description},{best_epoch},{best_fid}\n")


if __name__ == "__main__":
    main()
