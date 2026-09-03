import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import img_as_ubyte, io
from skimage.transform import resize
from sklearn import cluster
from utils import transform_mOT, transform_mPOT


np.random.seed(1)
torch.manual_seed(1)

parser = argparse.ArgumentParser(description="CT")
parser.add_argument("--m", type=int, default=10, metavar="N", help="input batch size for training (default: 100)")
parser.add_argument("--k", type=int, default=10, metavar="N", help="input num batch for training (default: 200)")
parser.add_argument("--T", type=int, default=5000, metavar="N", help="Num Interations")
parser.add_argument("--source", type=str, metavar="N", help="Source")
parser.add_argument("--target", type=str, metavar="N", help="Target")
parser.add_argument("--cluster", action="store_true", help="Use clustering")
parser.add_argument("--load", action="store_true", help="Load precomputed")
parser.add_argument("--palette", action="store_true", help="Show color palette")
args = parser.parse_args()


n_clusters = 3000
name1 = args.source  # path to images 1
name2 = args.target  # path to images 2
source = img_as_ubyte(io.imread(name1))
target = img_as_ubyte(io.imread(name2))
reshaped_target = img_as_ubyte(resize(target, source.shape[:2]))
name1 = name1.replace("/", "")
name2 = name2.replace("/", "")
if args.cluster:
    X = source.reshape((-1, 3))  # We need an (n_sample, n_feature) array
    source_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init=4, batch_size=100)
    source_k_means.fit(X)
    source_values = source_k_means.cluster_centers_.squeeze()
    source_labels = source_k_means.labels_

    # create an array from labels and values
    # source_compressed = np.choose(labels, values)
    source_compressed = source_values[source_labels]
    source_compressed.shape = source.shape

    vmin = source.min()
    vmax = source.max()

    # original image
    plt.figure(1, figsize=(5, 5))
    plt.title("Original Source")
    plt.imshow(source, vmin=vmin, vmax=256)

    # compressed image
    plt.figure(2, figsize=(5, 5))
    plt.title("Compressed Source")
    plt.imshow(source_compressed.astype("uint8"), vmin=vmin, vmax=vmax)
    os.makedirs("npzfiles", exist_ok=True)
    with open("npzfiles/" + name1 + "source_compressed.npy", "wb") as f:
        np.save(f, source_compressed)
    with open("npzfiles/" + name1 + "source_values.npy", "wb") as f:
        np.save(f, source_values)
    with open("npzfiles/" + name1 + "source_labels.npy", "wb") as f:
        np.save(f, source_labels)
    np.random.seed(0)

    X = target.reshape((-1, 3))  # We need an (n_sample, n_feature) array
    target_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init=4, batch_size=100)
    target_k_means.fit(X)
    target_values = target_k_means.cluster_centers_.squeeze()
    target_labels = target_k_means.labels_

    # create an array from labels and values
    target_compressed = target_values[target_labels]
    target_compressed.shape = target.shape

    vmin = target.min()
    vmax = target.max()

    # original image
    plt.figure(1, figsize=(5, 5))
    plt.title("Original Target")
    plt.imshow(target, vmin=vmin, vmax=256)

    # compressed image
    plt.figure(2, figsize=(5, 5))
    plt.title("Compressed Target")
    plt.imshow(target_compressed.astype("uint8"), vmin=vmin, vmax=vmax)

    with open("npzfiles/" + name2 + "target_compressed.npy", "wb") as f:
        np.save(f, target_compressed)
    with open("npzfiles/" + name2 + "target_values.npy", "wb") as f:
        np.save(f, target_values)
    with open("npzfiles/" + name2 + "target_labels.npy", "wb") as f:
        np.save(f, target_labels)
else:
    with open("npzfiles/" + name1 + "source_compressed.npy", "rb") as f:
        source_compressed = np.load(f)
    with open("npzfiles/" + name2 + "target_compressed.npy", "rb") as f:
        target_compressed = np.load(f)
    with open("npzfiles/" + name1 + "source_values.npy", "rb") as f:
        source_values = np.load(f)
    with open("npzfiles/" + name2 + "target_values.npy", "rb") as f:
        target_values = np.load(f)
    with open("npzfiles/" + name1 + "source_labels.npy", "rb") as f:
        source_labels = np.load(f)

k = args.k
m = args.m
iter = args.T
os.makedirs("npzfiles", exist_ok=True)

def npz_path(tag):
    return "npzfiles/{}_{}_to_{}_m{}_k{}_T{}.npy".format(tag, name1, name2, m, k, iter)

def load_npy(tag):
    with open(npz_path(tag), "rb") as f:
        return np.load(f)

def save_npy(tag, arr):
    with open(npz_path(tag), "wb") as f:
        np.save(f, arr)

# Load mOT results (compute together to avoid running transform_mOT twice)
if os.path.exists(npz_path("mOT")) and os.path.exists(npz_path("mOTcluster")):
    mOT = load_npy("mOT")
    mOTcluster = load_npy("mOTcluster")
else:
    mOTcluster, mOT = transform_mOT(source_values, target_values, source_labels, source, k=k, m=m, iter=iter)
    save_npy("mOT", mOT)
    save_npy("mOTcluster", mOTcluster)

# Load or compute m-POT with mass=0.9
if os.path.exists(npz_path("mPOT09")):
    mPOT09 = load_npy("mPOT09")
else:
    _, mPOT09 = transform_mPOT(source_values, target_values, source_labels, source, k=k, m=m, mass=0.9, iter=iter)
    save_npy("mPOT09", mPOT09)

# Load or compute m-POT with mass=0.99
if os.path.exists(npz_path("mPOT099")):
    mPOT099 = load_npy("mPOT099")
else:
    _, mPOT099 = transform_mPOT(source_values, target_values, source_labels, source, k=k, m=m, mass=0.99, iter=iter)
    save_npy("mPOT099", mPOT099)

# 5-panel visualization: Source | m-OT | m-POT s=0.9 | m-POT s=0.99 | Target
fig, ax = plt.subplots(1, 5, figsize=(15, 5))
ax[0].imshow(source)
ax[1].imshow(mOT)
ax[2].imshow(mPOT09)
ax[3].imshow(mPOT099)
ax[4].imshow(reshaped_target)
ax[0].set_title("Source", fontsize=14)
ax[1].set_title("m-OT", fontsize=14)
ax[2].set_title("m-POT s=0.9", fontsize=14)
ax[3].set_title("m-POT s=0.99", fontsize=14)
ax[4].set_title("Target", fontsize=14)
for i in range(5):
    ax[i].get_yaxis().set_visible(False)
    ax[i].get_xaxis().set_visible(False)

plt.tight_layout()
plt.subplots_adjust(left=0, right=1, top=0.88, bottom=0.01, wspace=0, hspace=0.145)

os.makedirs("images/results", exist_ok=True)
tag = "{}_to_{}_m{}_k{}_T{}".format(name1, name2, m, k, iter)
plt.savefig("images/results/figure16_{}.png".format(tag), dpi=150, bbox_inches="tight")
io.imsave("images/results/mOT_{}.bmp".format(tag), mOT)
io.imsave("images/results/mPOT09_{}.bmp".format(tag), mPOT09)
io.imsave("images/results/mPOT099_{}.bmp".format(tag), mPOT099)
plt.show()
