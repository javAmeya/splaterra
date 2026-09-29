# Direct 3D Anchor Loss to the Point-Cloud Prior

Depth supervision only constrains a Gaussian's position along the sampled
camera's viewing ray, so drift perpendicular to that ray goes uncorrected.
Anchor loss was added to close that gap: a direct, view-independent 3D
penalty expected to keep Gaussians closer to the trustworthy point-cloud
prior than depth supervision achieves alone.

Implementation: `tinysplat/tinysplat/gaussian_model.py` (`anchor_loss`, `xyz_init`).

## 1. Formulation

Let $x_g^{(0)} \in \mathbb{R}^3$ denote Gaussian $g$'s **birth position** —
the point-cloud point it was created from, tracked in `self.xyz_init` and
held fixed (non-trainable, no gradient) for that Gaussian's entire lifetime.
Let $x_g \in \mathbb{R}^3$ denote its current, actively-optimized position
(`self.xyz`). The anchor loss over the live population $\mathcal{G}$ (size
$G = |\mathcal{G}|$ at the current iteration) is

$$\mathcal{L}_{\text{anchor}} = \frac{1}{G}\sum_{g \in \mathcal{G}} \lVert x_g - x_g^{(0)} \rVert_2^2$$

a mean squared Euclidean displacement, added to the total training loss
with weight $\lambda_{\text{anchor}}$ (`opt.anchor_loss_weight`):

$$\mathcal{L}_{\text{total}} = (1-\lambda_{\text{dssim}})\,\mathcal{L}_1 + \lambda_{\text{dssim}}\,(1-\text{SSIM}) + w_{\text{depth}}(s)\,\mathcal{L}_{\text{depth}} + \lambda_{\text{aniso}}\,\mathcal{L}_{\text{aniso}} + \lambda_{\text{anchor}}\,\mathcal{L}_{\text{anchor}}$$

Unlike the photometric terms, $\mathcal{L}_{\text{anchor}}$ does not depend
on which camera $k_s$ is sampled this iteration — it is computed and its
gradient applied on **every** inner view, the same way `anisotropy_loss` is
(`train.py`: both are added to `loss` unconditionally, before the
per-view-conditional depth term).

This is a raw sum-of-squared-distance, **not normalized by scene scale** —
so $\lambda_{\text{anchor}} = 1.0$ (N5's value) is not a calibrated
"fraction of displacement," just an empirically-swept starting point (see
§5).

## 2. Birth-position lifecycle

$x_g^{(0)}$ must survive exactly as long as $g$ does, including through
densification and pruning:

- **Initialization** (`create_from_pcd`): $x_g^{(0)} \leftarrow x_g(0)$ for
  every original point-cloud point — birth position equals starting
  position exactly, so $\mathcal{L}_{\text{anchor}} = 0$ at iteration 0.
- **Pruning** (`_prune_points`): `self.xyz_init = self.xyz_init[valid]` —
  filtered by the same boolean mask as every other per-Gaussian tensor, so
  indices stay aligned.
- **Clone** (`_densify_and_clone`): a clone is a straight duplicate of an
  existing Gaussian at its current position. The clone **inherits its
  parent's own $x_g^{(0)}$** (`new_xyz_init = self.xyz_init[selected]`),
  not its current position — clone and parent are tethered to the *same*
  original point-cloud neighborhood, reflecting that they represent the
  same underlying prior evidence, now split into two Gaussians.
- **Split** (`_densify_and_split`): an oversized Gaussian divides into
  `n_split` (default 2) smaller children at perturbed positions. Each child
  also inherits the **parent's** $x_g^{(0)}$, repeated per child
  (`new_xyz_init = self.xyz_init[selected].repeat(n_split, 1)`) — same
  reasoning as clone.
- **Checkpointing**: `xyz_init` is saved/restored alongside every other
  model tensor (`capture`/`restore`), with a fallback for older checkpoints
  predating this mechanism (`state.get("xyz_init")`, `None` if absent —
  `anchor_loss()` returns a zero tensor in that case rather than erroring).

Net effect: at any point in training, however many generations of
clone/split a given Gaussian is descended from, $x_g^{(0)}$ still points to
*some* original point-cloud location in the same local neighborhood — the
tether is never lost, only ever inherited.

## 3. Interaction with depth supervision

Both mechanisms pull toward the same underlying prior but along different
axes, and are complementary rather than redundant:

| | constrains | independent of camera? |
|---|---|---|
| Depth supervision | depth *along the current ray only* | no — only active for the sampled view |
| Anchor loss | full 3D displacement | yes — active on every view |

A Gaussian could in principle satisfy depth supervision perfectly (correct
distance from every camera that happens to render it) while still having
drifted sideways within the constraint surface each ray defines — anchor
loss is the term that catches that case. Empirically, N1-N10's factorial
sweep found the combination (`D+A`, i.e. N5) outperformed either alone
(`docs/experiment_log.md`, "N1-N8" section: N5 ssim 0.834 vs. N2's
depth-only 0.674 and N3's anchor-only 0.622 at 60k).

## 4. Weight sensitivity — non-monotonic, not yet fully characterized

$\lambda_{\text{anchor}}$ was swept across $\{0.3, 1.0, 1.5, 2.0\}$ at 60k
(N9/N10, N5, N12, N13) and $\{1.0, 1.25, 1.5, 1.75, 2.0\}$ at 120k
(N11, N17-N20):

| weight | 60k ssim | 120k ssim |
|---|---|---|
| 0.3 | worse than 1.0 (N9/N10, not tabulated in detail) | — |
| 1.0 | **0.834** (N5) | 0.552 (N11) |
| 1.25 | — | 0.668 (N17) |
| 1.5 | 0.647 (N12, a dip) | **0.716** (N18) |
| 1.75 | — | 0.511 (N19, worst of the 120k batch) |
| 2.0 | 0.819 (N13) | 0.707 (N20) |

At 60k, the curve is **not monotonic**: 1.0 and 2.0 are both strong and
nearly tied, with a real dip at 1.5 between them — more a plateau at the
extremes with a trough in the middle than a single-peaked curve. At 120k
the picture is noisier still (1.75 is the worst point in the whole set,
flanked by two better values) — see `docs/experiment_log.md`'s
"N17/N18/N19/N20" and "Metric-vs-visual-verdict correspondence" sections
for the full discussion, including why these numbers alone shouldn't be
fully trusted without a visual check (single-step summary snapshots can be
noisy — see `docs/depth_loss_spike_notes` discussion inline in the log).
**Not yet characterized with enough resolution to claim a true optimum** —
the swept points are too coarse and too confounded with duration to fit a
clean curve.

## 5. Open directions

- **Scale-normalized variant**: $\mathcal{L}_{\text{anchor}}$ is currently
  a raw sum-of-squared-distance in world units, not normalized by scene
  scale (`cameras_extent` or `median_nn_distance`, both already computed
  elsewhere in this codebase — see `docs/point_density_extent_method.md`).
  A normalized version might make $\lambda_{\text{anchor}}$ transferable
  across scenes of different physical scale, which the current raw form
  is not.
- **Per-Gaussian confidence weighting**: every point-cloud point is
  currently anchored with equal weight regardless of how confident LoGeR
  was about that specific point (`loger_to_colmap.py` already computes and
  exports a per-point confidence value for depth supervision's mask —
  currently unused by anchor loss).
- **Finer weight sweep**: the 60k and 120k sweeps above used coarse
  0.25-0.5 increments; the true response curve (especially the 1.5 dip and
  the 1.75 collapse) has not been resolved at finer resolution.
