# Mini-batch Keypoint-Guided Relative Optimal Transport (m-KPOT)

A mini-batch OT framework that combines two ideas:

- **Mini-batch (Partial / Unbalanced) OT** (Nguyen et al., ICML 2022) — solves
  many small per-batch OT problems instead of a single huge one, and optionally
  caps the transported mass (m-POT) or relaxes the marginals (m-UOT) to absorb
  noisy or outlier transports.
- **Keypoint-Guided OT with Relation Preservation (KPG-RL)** (Gu et al.,
  NeurIPS 2022) — uses a small set of *keypoint anchors* to steer the
  transport plan towards structurally consistent matchings.

m-KPOT applies KPG-RL guidance *inside each mini-batch* so the OT plan benefits
from both the noise-robustness of partial / unbalanced OT and the structural
prior of keypoint relations.

---

## Which setting feeds which paper

| Setting | Flag | Paper |
|---|---|---|
| Balanced (m-KOT) | `--ot_type balanced` / `--method jdot` / `--method OT` | `Mini-batch_Keypoint-OT/mKOT-KPOT.tex` |
| Partial (m-KPOT) | `--ot_type partial` / `--method jpmbot` / `--method POT` | `Mini-batch_Keypoint-OT/mKOT-KPOT.tex` |
| Unbalanced (m-KUOT) | `--ot_type unbalanced` / `--method jumbot` / `--method UOT` | `Mini-batch_Keypoint-UOT/mKUOT.tex` |

One implementation serves all three — the mask, relation profiles, guiding matrix and blended
cost are identical, and only the inner marginal constraint changes (Sec. IV-B: guidance is
orthogonal to the marginal constraint).

**Transported mass `s`** is fixed, not tuned; only the guidance weight `α` is swept:
`0.65` on Office-Home, VisDA-2017, Office-Home PDA and DeepGM, with **one exception** —
Digits m-KPOT uses a per-task `s` of `0.85 / 0.90 / 0.80` for
SVHN→MNIST / USPS→MNIST / MNIST→USPS (set in `DeepDA/digits/sh/train_KPG_mPOT.sh`).

---

## Method

There is **one** implementation: Section IV of the manuscript. (The pre-2026-07-29 "Option A"
code — no mask, per-batch virtual centroid anchors, off-the-shelf POT — has been removed, along
along with the `--kpg_variant` switch that selected between the two, and the
`--tau_s / --tau_t / --target_kp / --n_anchor_samples` flags only it needed.  There is no
variant argument any more: `--use_kpg` means Sec. IV.)

The mathematics is **inlined in each task's own implementation**, matching how this repository
already keeps its per-task helpers self-contained — no shared module, no import shims. The block
is delimited by a banner comment (`m-KUOT / m-KOT / m-KPOT -- Sec. IV of the paper, inlined`) in:

| File | Contains |
|---|---|
| [`src/DeepDA/office/train.py`](src/DeepDA/office/train.py) | full core + `KeypointBank` + keypoint-first batching |
| [`src/PartialDA/run_mKPOT.py`](src/PartialDA/run_mKPOT.py) | full core + `KeypointBank` + `inject_keypoints` |
| [`src/DeepDA/digits/methods.py`](src/DeepDA/digits/methods.py) | full core + `KeypointBank` + `inject_keypoints` |
| [`src/DeepGM/kpg_ot.py`](src/DeepGM/kpg_ot.py) | core (NumPy only); the generators import from here |

The four copies are byte-identical in behaviour — verified by extracting each inlined block and
running the same numerical suite against it (mask structure, `G ∈ [0, log 2]`, zero mass outside
the mask, Eq. (13) residual < 1e-12, `α = 1` ignoring `G` bit-exactly, and the `k = m` guard).
**If you change one, change all four.**

### Not supported

- **BoMb + keypoint guidance.** Section IV's estimator is the uniform average of Eq. (8); the
  hierarchical batch-of-mini-batches scheme is a different estimator, so `--use_bomb` (or
  `fit_bomb` / `fit_bomb2`) together with `--use_kpg` is refused. The `*_BoMb*.sh` scripts
  therefore run the **unguided** BoMb baseline.
- **MNIST generative + keypoint guidance.** That model has no discriminator (OT runs in
  flattened pixel space), so the plan-mined keypoint pairs used for CelebA / CIFAR-10 do not
  transfer, and MNIST is not a reported experiment. `--use-kpg` is refused there.

### The method — Section IV

Fix `k` keypoint pairs `K = {(x̂_1, ŷ_1), …, (x̂_k, ŷ_k)}` with annotated correspondence.
Every mini-batch places them in its **first k slots** and samples only the remaining
`m − k` points (Eq. 5).  On each batch we solve

```
   KUOT = min_{T ∈ Pi_M}  <C~, T> + tau*KL(T 1 || u) + tau*KL(T^T 1 || u)      (11)
```

over the **masked** feasible set `Pi_M = {T >= 0 : T = M (*) T}` (Eq. 7), with

```
   M    = [[ I_k , 0 ], [ 0 , 1_{(m-k)x(m-k)} ]]                              (6)
   r_{i,i'} = softmax_{i'}( -c_{i,i'} / (rho * max c) )                       (8)
   G_{i,j}  = JSD( r^x_i , r^y_j )        in [0, log 2]                       (9)
   C~       = alpha * C/max(C) + (1 - alpha) * G                              (10)
```

The mask makes keypoint `x_c` transport **only** to `ŷ_c` and vice versa; the all-ones
lower-right block leaves the `m − k` sampled points free to match one another.  Solving
Eq. (11) entropically gives `K = M (*) exp(-C~/eps)` and the scaling iterations

```
   f <- ( u / (K g)   )^( tau / (tau + eps) )                                 (15)
   g <- ( u / (K^T f) )^( tau / (tau + eps) )        T = diag(f) K diag(g)
```

implemented in the log domain for stability.  The exponent `tau/(tau+eps)` is the only
difference from balanced Sinkhorn; as `tau -> inf` it tends to 1 and the classical
algorithm is recovered.

### Three consequences of the mask

1. **Keypoints are real data points, not centroids.**  The mask acts on rows and columns of
   `T`, so a keypoint must occupy a batch slot — a class *centroid* is not a sample and
   cannot.  Wherever the paper says "centroid" the code therefore uses the class **medoid**
   (the sample nearest the centroid).
2. **Keypoints are fixed across mini-batches.**  Their dataset indices are chosen once,
   before training, and reused in every batch.  Their *embeddings* still move as the encoder
   trains — the pairs are fixed, not their coordinates.
3. **k must be < m.**  See the warning below.

### Keypoint-selection strategies (Sec. V-A-3 ablation)

Source side is always the class medoid (source labels are available by definition); only the
target side varies, so the spread between the three isolates target-keypoint quality.

| `--kp_strategy` | Target keypoint | Labels used | Role |
|---|---|---|---|
| `centroid` | medoid of the **true** target class | GT target labels | **oracle upper bound** — not deployable |
| `random`   | random sample of the **pseudo**-class | pseudo-labels | **practical / deployable** |
| `farthest` | pseudo-class sample farthest from its centroid | pseudo-labels | adversarial lower bound |

Expected ordering: `centroid` ≥ `random` ≥ `farthest`.

> ### ⚠️ `k` must be `< m`
> `check_batch_feasibility` **raises** when `k == m`: every slot would be a keypoint, the mask
> collapses to the identity and no free transport remains.  Office-Home closed-set hits this at
> the original `m = 65` (one keypoint per class over 65 classes ⇒ `k = 65`), so it now runs at
> **`m = 130`** — 65 reserved slots + 65 free, and 2 source samples per class via the balanced
> sampler.  Every configuration currently in the scripts passes: Digits 500/10, Office-Home
> 130/65, VisDA-2017 72/12, Office-Home PDA 65/25, DeepGM 200/5 and 100/5.  If you need a
> smaller batch, cut `k` instead with `--kp_n_classes`.

### How many scaling iterations

Convergence is set by `p = tau/(tau+eps)` — the closer to 1, the slower.  Max residual of the
first-order condition Eq. (13), `eps = 0.01`:

| configuration | `p` | N=200 | N=500 | N=1000 | N=2000 |
|---|---|---|---|---|---|
| Digits, `tau=1.0`  | 0.9901 | 2.2e-03 | 5.6e-06 | 2.6e-10 | 9.9e-13 |
| VisDA, `tau=0.3`   | 0.9677 | 3.9e-07 | 2.8e-13 | 2.8e-13 | 2.8e-13 |
| PDA, `tau=0.06`    | 0.8571 | 4.7e-14 | 4.7e-14 | 4.7e-14 | 4.7e-14 |

`--kuot_iters` therefore defaults to **1000**, not a few hundred: at `tau = 1.0` (Digits)
N = 200 leaves a 2e-3 residual, i.e. a visibly sub-optimal plan.  A `tol` early-exit keeps
the easy configurations cheap.

### DeepGM (no labels)

A keypoint pair is *(a fixed real image, a fixed latent code `z`)*, mined once from an
initial transport plan.  The generated member's features move as the generator trains,
exactly as the DA keypoints' embeddings move as the encoder trains.

---

## Project structure

```
Mini-batch_KPG-RL_OT/
├── README.md                       — this file
├── requirements.txt
├── data/                           — datasets (Office-Home, Office-31, digits, ...)
├── figures/                        — motivation / illustration figures
└── src/
    ├── DeepDA/
    │   ├── office/                 — Office-Home / Office-31 / VisDA
    │   │   ├── train.py            — mini-batch KPG-OT trainer
    │   │   └── sh/                 — train_{home,office}_KPG_{mOT,BoMbOT}.sh
    │   └── digits/                 — SVHN ↔ MNIST ↔ USPS
    │       ├── methods.py          — DigitsDA trainer (fit / fit_bomb / fit_bomb2)
    │       ├── train_digits.py
    │       ├── cfg.py
    │       └── sh/                 — train_KPG_{mOT,mUOT,mPOT,BoMbOT,BoMbUOT}.sh
    ├── DeepGM/                     — CIFAR-10 / CelebA / MNIST generative models
    │   ├── kpg_ot.py               — KPG-RL OT solver (two-pass, unsupervised)
    │   ├── {Cifar,Celeba,Mnist}_generator.py
    │   ├── main_{cifar,celeba,mnist}.py
    │   └── sh/                     — run_KPG_{cifar,celeba,mnist}.sh
    └── PartialDA/                  — Partial Domain Adaptation on Office-Home
        ├── run_mKPOT.py
        └── sh/                     — train_home_KPG_{mOT,mUOT,mPOT}.sh
```

---

## Requirements

```bash
# Python 3.12.13
numpy==2.4.4
scipy==1.17.1
matplotlib==3.10.8
geomloss==0.2.6
pot==0.9.6
cvxpy==1.8.2
torch==2.11.0
torchvision==0.26.0
tqdm==4.67.3
imageio
```

(See `requirements.txt`.  Newer versions generally work — the code has been
tested with PyTorch 2.x as well.)

---

## Data preparation

| Dataset | Where to put it |
|---|---|
| Office-Home | `data/office-home/images/<domain>/<class>/<img>.jpg` |
| Office-31   | `data/office/<domain>/<class>/<img>.jpg`            |
| SVHN, MNIST, USPS | `data/{svhn,mnist,usps}/`                     |
| CelebA      | `data/celeba/` (cropped 64×64)                      |
| CIFAR-10    | `data/cifar10/`                                     |

For Office-Home / Office-31 you can also point the code at an external
directory by setting the environment variable `OFFICE_HOME_IMAGES_ROOT` (or
`OFFICE31_IMAGES_ROOT`) — the data-list resolver in
[src/DeepDA/office/train.py](src/DeepDA/office/train.py) will pick it up.

---

## Running experiments

Each experiment ships with shell scripts that fix the hyperparameters used in
the paper.  All scripts take a GPU id as the first argument (default `0`) and
support common knobs via environment variables.

### DeepDA — Office-Home (12 transfers)

```bash
cd src/DeepDA/office

# Mini-batch KPG-OT (balanced)
bash sh/train_home_KPG_mOT.sh 0

# BoMb-OT variant
bash sh/train_home_KPG_BoMbOT.sh 0
```

The no-KPG baseline (m-OT / m-UOT / m-POT without keypoint guidance) lives in the
sibling `../Baselines/Mini-batch-OT/DeepDA/office/` and is run from there
(`bash sh/home_mOT.sh 0`).

### DeepDA — digits (SVHN → MNIST, USPS ↔ MNIST)

```bash
cd src/DeepDA/digits

# Standard mOT-style training with KPG guidance
bash sh/train_KPG_mOT.sh 0            # m-KPOT balanced
bash sh/train_KPG_mUOT.sh 0           # m-KUOT unbalanced
bash sh/train_KPG_mPOT.sh 0           # m-KPOT partial

# BoMb (hierarchical k×k OT between mini-batches)
bash sh/train_KPG_BoMbOT.sh 0
bash sh/train_KPG_BoMbUOT.sh 0
```

### DeepGM — generative models on CIFAR-10 / CelebA / MNIST

```bash
cd src/DeepGM
bash sh/run_KPG_cifar.sh 0
bash sh/run_KPG_celeba.sh 0
bash sh/run_KPG_mnist.sh 0
```

### Partial DA — Office-Home 65/25

```bash
cd src/PartialDA
bash sh/train_home_KPG_mOT.sh  0      # m-KOT  + KPG
bash sh/train_home_KPG_mUOT.sh 0      # m-KUOT + KPG
bash sh/train_home_KPG_mPOT.sh 0      # m-KPOT + KPG  (recommended for PDA)
```

---

## Key CLI flags (shared across experiments)

Shared:

| Flag | Meaning | Default |
|---|---|---|
| `--ot_type {balanced,unbalanced,partial}` (DA) / `--method {OT,UOT,POT}` (DeepGM) | inner OT type | `balanced` / `OT` |
| `--use_kpg` (DA) / `--use-kpg` (DeepGM) | enable keypoint guidance | off |
| `--alpha` | blending in Eq. (10), `C~ = alpha*Cbar + (1-alpha)*G` | `0.5–0.9` |
| `--mass` | transported mass (partial only) | task-specific |
| `--tau` | marginal penalty (unbalanced only) | task-specific |
| `--epsilon` | entropic OT regularisation | `0` (exact LP) |
| `--k` | number of mini-batches per gradient step | `1` |
| `--n_shared_classes` | (PDA only) restrict keypoints to the shared classes | `None` |

Keypoint-guidance parameters (Section IV):

| Flag | Meaning | Default |
|---|---|---|
| `--kp_strategy {centroid,random,farthest}` | keypoint-selection strategy (Sec. V-A-3) | `random` |
| `--kp_per_class` | keypoint pairs per class; `k = kp_per_class × classes`, must satisfy `k < m` | `1` |
| `--kp_n_classes` | restrict keypoints to the first N classes (also the way to keep `k < m`) | all |
| `--rho` | dimensionless relation temperature of Eq. (8); softmax scale `rho·max(c)` | `0.1` |
| `--kp_metric {euclidean,sqeuclidean}` | ground metric for Eq. (8) | `euclidean` |
| `--kp_probe` | samples per domain encoded once to choose the fixed keypoints | `4096` |
| `--kp_replace` | sample the `m−k` free slots with replacement | off |
| `--kuot_eps` | entropic regularisation of Eq. (12) when `--epsilon` is 0 | `0.01` |
| `--kuot_iters` | scaling iterations `N` in Alg. 1 | `1000` |

> `--rho` replaces the removed `--tau_s`/`--tau_t` and is **not** numerically comparable to
> them: the old code used `exp(-2 · normalised SQUARED distance / tau)`, whereas Eq. (8) uses
> `exp(-c / (rho·max c))` on the raw ground metric.  Tune it rather than carrying `0.1` over
> on the assumption that it means the same thing.

---

## Ablation: how to compare the three target-keypoint strategies

The default `--kp_strategy random` is the realistic, label-free setting.  To quantify how much
of the gain comes from *good* keypoint choice, sweep all three strategies on the same
configuration:

```bash
# Office-Home Art → Clipart, all three target-keypoint strategies
for kp in centroid random farthest; do
    python train.py … --use_kpg --alpha 0.5 --kp_strategy $kp \
                       --output_dir A2C_kpg_${kp}
done
```

You should observe:  centroid (oracle) ≥ random ≥ farthest.

---

## Comparison with baselines

The reference baselines this codebase is designed to compare against live in
the sibling directory `../Baselines/`:

| Baseline | Reference |
|---|---|
| `Mini-batch-OT` (m-OT, m-UOT, m-POT, BoMb-OT) | Nguyen et al., *On Transportation of Mini-batches: A Hierarchical Approach* and *Improving Mini-batch Optimal Transport via Partial Transportation*, ICML 2022 |
| `PPOT` (m-PPOT) | Yang et al., *Prototypical Partial OT for Universal DA*, AAAI 2023 |
| `WARMPOT` | Naram et al., *Theoretical Performance Guarantees for Partial Domain Adaptation via Partial OT*, ICML 2025 |

The Mini-batch-OT baseline scripts also expose a `--use_mirror_sinkhorn` flag
(see `../Baselines/Mini-batch-OT/`) so the inner OT solver can be swapped for
Mirror Sinkhorn (Ballu & Berthet, ICML 2023) for an apples-to-apples bias-free
solver comparison.

---

## References

If you use this code, please cite the underlying methods:

- Nguyen, K., Nguyen, D., Pham, T., and Ho, N.  *On Transportation of
  Mini-batches: A Hierarchical Approach* (ICML 2022).
- Nguyen, K., Nguyen, D., Nguyen, T., Pham, T., and Ho, N.  *Improving
  Mini-batch Optimal Transport via Partial Transportation* (ICML 2022).
- Gu, X., Yang, Y., Sun, J., Xu, Z.  *Keypoint-Guided Optimal Transport with
  Applications in Heterogeneous Domain Adaptation* (NeurIPS 2022).

For the Mirror Sinkhorn baseline solver:

- Ballu, M., Berthet, Q.  *Mirror Sinkhorn: Fast Online Optimization on
  Transport Polytopes* (ICML 2023).
