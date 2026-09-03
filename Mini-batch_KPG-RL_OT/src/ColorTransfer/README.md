# Color Transfer with m-KUOT

Colour transfer with **mini-batch keypoint-guided unbalanced optimal transport**
(paper, Sec. IV), following the Mini-batch-OT template pipeline
(`Baselines/Mini-batch-OT/ColorTransfer`): both images are compressed to a
3000-colour palette by mini-batch k-means, and the source palette is mapped onto
the target palette by a mini-batch barycentric projection.

## Method

At each of the `T` steps, `k` mini-batches of size `m` are drawn per side and every
of the `k x k` sub-batch pairs is solved independently. The inner problem is the
Sec. IV keypoint-guided problem:

- **Keypoints (unsupervised).** Colour transfer has no labels, so the DeepGM
  protocol applies: a preliminary UNGUIDED plan is solved once on a large random
  batch (`--init_m` per side) and its `--n_kp` highest-mass entries — no row or
  column reuse — define fixed (source colour, target colour) pairs. They occupy
  the first `n_kp` slots of every mini-batch on both sides (Eq. 5) for the whole
  run.
- **Mask (Eq. 6).** Pins every keypoint to its partner; the remaining
  `m - n_kp` colours stay free.
- **Guidance (Eqs. 8-10).** Every free colour gets a relation profile (softmax,
  temperature `--rho`) to the keypoints of its own image; the Jensen-Shannon
  divergence between profiles forms the guiding matrix `G`, blended into the
  max-normalised task cost as `C~ = alpha * C + (1 - alpha) * G`.
- **Solver.** The masked unbalanced problem is solved by the scaling iterations of
  Eq. (15) (`--tau`, `--reg`, `--n_sink`). Balanced (`m-KOT`) and partial
  (`m-KPOT`) variants use the identical mask and guiding matrix (Sec. IV-B).

The unguided baselines m-OT / m-UOT / m-POT are the `alpha = 1`, all-ones-mask
corner of the same pipeline, so guided-vs-unguided differences are attributable to
keypoint guidance alone. The template's original baseline implementations are kept
verbatim in `utils.py` for cross-checking.

**Reconstruction.** Because the keypoint slots appear in every sub-batch (Sec.
IV-C), the raw accumulate-and-rescale of the template would inflate the keypoint
clusters by their appearance count; all runs therefore use the mass-normalised
barycentric projection `colour(i) = sum (T @ t)_i / sum (T 1)_i`, which for the
balanced unguided case equals the template output up to a global rescale.

**Target-keypoint strategies** (Sec. V's three variants, transposed to the
unlabeled setting; all three share the same source keypoints):

| strategy | target member of each pair |
|---|---|
| `c` | the preliminary-plan partner (highest-confidence correspondence) |
| `r` | a random target palette colour |
| `f` | the target colours farthest from the preliminary batch mean (deliberately peripheral) |

## Terminologies

- `--k` : number of mini-batches, `--m` : size of mini-batches, `--T` : steps
- `--n_kp` : number of keypoint pairs, `--alpha` : guidance weight
  (smaller = stronger guidance), `--rho` : relation temperature (Eq. 8)
- `--tau` : marginal KL parameter, `--reg` : entropic regularisation,
  `--mass` : transported mass for partial runs
- `--cluster` : run the k-means compression (once per image pair)
- `--run` : `prep` | `all` | comma list of
  `mOT, mUOT, mPOT09, mPOT099, mKUOT-{c|r|f}:{alpha}, mKOT-..., mKPOT-...`
- `--figure` : assemble figures + metrics from cached runs; `--palette` : also
  save the mined keypoint colour pairs

## Evaluation

`images/results/CT_metrics_*.csv` reports three numbers per run (all in RGB scaled
to `[0,1]^3`, lower is better):

1. **`W2sq_to_target_palette`** — exact squared Wasserstein-2 distance between the
   transferred palette and the target palette (pixel-count weighted): endpoint
   fidelity of the recolouring.
2. **`map_RMSE_to_fulldata_same_regime`** — pixel-weighted RMSE between the
   mini-batch barycentric map and the barycentric map of the FULL `n x n` problem
   of the same regime (balanced / unbalanced / partial, same solver conventions).
   Colour transfer is small enough to solve the full problem, so this is the
   direct mini-batch-vs-full-data plan-fidelity measurement that the paper's
   Discussion lists as untested. Note the unbalanced full reference inherits the
   entropic blur of `eps = 0.01` at `n = 3000`.
3. **`crossbatch_std`** — pixel-weighted mean of the mass-weighted standard
   deviation of each palette entry's per-batch barycentric image across all
   sub-batch pairs. This is the mismatching signal itself: it is zero exactly
   when every mini-batch maps the colour to the same place, and the paper's
   cross-batch-consistency reading predicts keypoint guidance lowers it.

Figures: `CT_family_*` (family comparison incl. the full-data OT reference),
`CT_alpha_grid_*` (strategy x alpha), `CT_alpha_sensitivity_*` (paper-Fig.-1-style
sweep of endpoint fidelity and cross-batch consistency), `CT_keypoints_*` (mined
pairs).

## To reproduce

```bash
conda activate mkpg-ot
python main.py --source images/s1.bmp --target images/t1.bmp --cluster --run prep
# runs are cached; parallelise freely, e.g. one process per run spec
python main.py --source images/s1.bmp --target images/t1.bmp --run all
python main.py --source images/s1.bmp --target images/t1.bmp --run all --figure --palette
```

Defaults (`m=100, k=4, T=1000, n_kp=5, rho=0.1, tau=1.0, reg=0.01`) follow the
paper's DeepGM configuration; the runtime per run spec is ~5-10 min on CPU.
