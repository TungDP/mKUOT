"""
Precompute FID statistics for the CIFAR-10 test set and save to
fid_stats_cifar_test.npz, which is required by main_cifar.py.

Usage:
    python make_fid_stats_cifar.py --datadir ./data
"""

import argparse
import os
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

from inception import InceptionV3


def get_activations_from_loader(dataloader, model, dims, device):
    model.eval()
    pred_arr = []
    for batch, _ in tqdm(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr.append(pred.squeeze(3).squeeze(2).cpu().numpy())
    return np.concatenate(pred_arr, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="./data", help="path to CIFAR-10 data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default="fid_stats_cifar_test.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(root=args.datadir, train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False
    )

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)

    print(f"Computing Inception activations for {len(dataset)} CIFAR-10 test images...")
    act = get_activations_from_loader(dataloader, model, 2048, device)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    np.savez(args.output, mu=mu, sigma=sigma)
    print(f"Saved FID stats to {args.output}  (mu shape: {mu.shape}, sigma shape: {sigma.shape})")


if __name__ == "__main__":
    main()
