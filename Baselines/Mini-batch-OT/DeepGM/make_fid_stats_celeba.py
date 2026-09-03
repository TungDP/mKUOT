"""
Precompute FID statistics for the CelebA test set and save to
fid_stats_celeba_test.npz, which is required by main_celeba.py.

Usage:
    python make_fid_stats_celeba.py --datadir ./data/celeba
"""

import argparse
import csv
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import Dataset
from tqdm import tqdm

from inception import InceptionV3


class CelebATestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir = os.path.join(root, "img_align_celeba")
        self.transform = transform
        with open(os.path.join(root, "list_eval_partition.csv")) as f:
            reader = csv.DictReader(f)
            self.filenames = [
                row["image_id"] for row in reader if int(row["partition"]) == 2
            ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.filenames[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def get_activations_from_loader(dataloader, model, device):
    model.eval()
    pred_arr = []
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr.append(pred.squeeze(3).squeeze(2).cpu().numpy())
    return np.concatenate(pred_arr, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="./data/celeba", help="path to CelebA data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", default="fid_stats_celeba_test.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CelebATestDataset(
        args.datadir,
        transform=transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ]),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)

    print(f"Computing Inception activations for {len(dataset)} CelebA test images...")
    act = get_activations_from_loader(dataloader, model, device)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    np.savez(args.output, mu=mu, sigma=sigma)
    print(f"Saved FID stats to {args.output}  (mu shape: {mu.shape}, sigma shape: {sigma.shape})")


if __name__ == "__main__":
    main()
