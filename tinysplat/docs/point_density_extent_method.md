# Point-Density-Based Extent for Densify/Prune Thresholds

3DGS's densify/prune thresholds normally scale with `cameras_extent`, how
far apart the training cameras were, which stays large on a long
walkthrough regardless of how fine the actual geometry is. This was tried
as a fix: derive the threshold from the point cloud's own local density
instead, expected to track real geometric scale rather than camera travel
distance. Tested once (arm N4), badly over-pruned, never used again.

Implementation: `tinysplat/tinysplat/gaussian_model.py` (`median_nn_distance`).

## 1. Formulation

At `create_from_pcd` time, for each point-cloud point $x_i$ (of $N$ total),
a $k$-nearest-neighbor query (`scipy.spatial.cKDTree`, $k=4$ so the
self-distance is included and dropped) gives the mean squared distance to
its 3 nearest neighbors — this is the *same* KNN query already computed
for per-Gaussian scale initialization, reused here at no extra cost.
`median_nn_distance` is the **median** (not mean, so a handful of
sparse-region outliers can't dominate it) of the per-point nearest-neighbor
distance across the whole cloud:

$$d_i = \sqrt{\frac{1}{3}\sum_{j=1}^{3} \lVert x_i - \text{NN}_j(x_i) \rVert^2}, \qquad d_{\text{median}} = \text{median}_i(d_i)$$

To use this as a threshold source, it's rescaled back up to an
"extent-equivalent" number via a single calibration constant, so the
existing $0.01 \times \text{extent}$ / $0.1 \times \text{extent}$ inner
multipliers throughout `gaussian_model.py` (clone/split boundary and
oversized-prune cutoff respectively — see `densify_and_prune`,
`_densify_and_clone`, `_densify_and_split`, `_prune_by_opacity_and_size`)
don't need re-deriving:

$$\text{extent}_{\text{density}} = d_{\text{median}} \times c, \qquad c = \texttt{point\_density\_extent\_scale} = 70.0$$

$c=70.0$ was calibrated **once**, against this specific scene's own
observed ratio $\text{cameras\_extent} / d_{\text{median}} \approx 2.84 /
0.04 \approx 71$ — a sane starting point for *this* scene, explicitly
documented as not a universal constant. `opt.use_point_density_extent`
(bool, `False` in every run including N5) selects which extent source
feeds `densify_and_prune`'s `extent` argument:

```python
densify_extent = (
    gaussians.median_nn_distance * opt.point_density_extent_scale
    if opt.use_point_density_extent and gaussians.median_nn_distance is not None
    else scene.cameras_extent
)
```

## 2. Empirical result — over-pruned badly

**Arm N4** (the only run to actually enable this): `max_scale` stayed
dramatically smaller than every other arm throughout training (0.047-0.063
vs. ~0.28-0.29 for N1/N2/N3, roughly a 5-6x difference) — not a modest
tightening, an order-of-magnitude stricter size ceiling. This showed up as
a real, visible quality cost, not just a number: **user's verdict, "N4 has
no details?? like its so sparse i can hardly see anything."** The
`max_scale` gap fully explains the complaint — the threshold was
suppressing legitimate large/flat-surface Gaussians along with whatever
actual floaters it was meant to catch, exactly the risk flagged when the
mechanism was first introduced ("could mean the threshold is too tight and
suppressing legitimate large flat surfaces" — confirmed).

**Root cause: $c=70.0$ was a back-of-envelope estimate, never actually
validated against the resulting `max_scale` distribution before committing
a full training run to it.** The calibration used the *ratio* of
`cameras_extent`/`median_nn_distance` at time zero, but that ratio doesn't
account for how the two quantities are actually *used* downstream inside
`densify_and_prune` (different multipliers, different roles in clone vs.
split vs. prune decisions) — matching the raw ratio doesn't guarantee
matching behavior.

## 3. Status: dropped, not re-attempted

N4 was scrapped and this mechanism has not been revisited (`M` was
explicitly excluded from N5-N8's later combinations —
`docs/experiment_log.md`: "drop M (median-extent) from any future N5-N8
launches"). `median_nn_distance` itself remains computed and stored (it's
essentially free, reusing an existing KNN query, and is harmless dead data
when `use_point_density_extent=False`), but the extent-selection mechanism
built on top of it has not been used since.

## 4. Open directions (if ever revisited)

- **Proper calibration pass before training**: binary-search
  `point_density_extent_scale` against the resulting `max_scale`
  distribution staying in the ~0.2-0.3 range other (working) arms show,
  rather than a one-shot ratio estimate — this was the flagged-but-skipped
  step that caused N4's failure.
- **Per-region rather than global**: a single scalar `median_nn_distance`
  for the whole scene may not suit a scene with genuinely varying point
  density (dense near-camera detail vs. sparse far background) — untried.
