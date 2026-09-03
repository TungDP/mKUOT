## Color Transfer (Color Transfer)

### Terminologies

--k : number of mini-batches

--m : the size of mini-batches

--T : the number of steps

--cluster: K-means clustering to compress images

--palette: show the color palette

--source: Path to the source image

### Mathematical Transformation

The color transfer is formulated as a Barycentric projection using Optimal Transport. Let $X_s$ be the source color distribution and $X_t$ be the target color distribution. To handle large images efficiently, the algorithm uses **Mini-batch Optimal Transport (mOT)**:

1. **Mini-batch Subsampling**: At each of the $T$ steps, the algorithm samples $k$ mini-batches of size $m$ from both the source and target images: $S_1, \dots, S_k$ and $T_1, \dots, T_k$.
2. **Pairwise Optimal Transport**: For every pair of mini-batches $(S_i, T_j)$, the optimal transport plan $P_{i,j}$ is computed by minimizing the transport cost $M = \text{dist}(S_i, T_j)$. Depending on the method (mOT, mUOT, mPOT), this plan is solved using exact Earth Mover's Distance, Unbalanced Sinkhorn, or Partial Wasserstein respectively.
3. **Barycentric Mapping**: The source points in $S_i$ are incrementally mapped towards the target points in $T_j$ using the local plan $P_{i,j}$. The total transformation for the source points is averaged over the $k \times k$ cross-combinations:
   $$ \hat{S}_i = \hat{S}_i + \frac{m}{k^2} P_{i,j} T_j $$
   *(Note: The $m$ factor corrects for the normalized probability mass $1/m$ used by the OT solver).*


To run Color Transfer experiments in the paper:
```
python main.py  --m=100 --T=10000 --source images/s1.bmp --target images/t1.bmp --cluster

```
For more detailed settings, please check them in our papers.