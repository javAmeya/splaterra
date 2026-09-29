# N5 Configuration

## 1. Experimental procedure

Runs 1-8 fixed a train/eval exposure mismatch, the single highest-leverage
change in the project. A COLMAP bundle-adjustment refinement raised PSNR
but looked worse on visual inspection, first sign PSNR wasn't trustworthy.

Pose correction (Run 9-10) cut parallax-duplicate artifacts sharply while
*lowering* PSNR, confirming the metric should be dropped. Retained in
every config since.

Run 34 found training running below the source video's native resolution.
Fixing it didn't resolve the blur alone, resolution wasn't the bottleneck.

N1-N10 (2026-09-27) tested three ways to keep the point-cloud prior
influencing Gaussians beyond initialization: depth supervision, anchor
loss, and a density-based prune threshold. The threshold over-pruned badly
and was dropped after one run. Depth supervision + anchor loss together
gave the best result of the round — N5.

Extending duration to 120k, on N5 and later across a range of anchor
weights, consistently hurt SSIM regardless of weight — attributed to
unreplenished pruning past the densify window. Sweeping anchor weight
around 1.0 gave a non-monotonic response (one intermediate value clearly
worse, neighbors fine) but nothing beat N5. Testing steeper decay, higher
init LR, and no decay against N5's existing schedule, all underperformed
it. Duration and LR schedule were fixed at N5's values on this basis.

## 2. Configuration

60,000 iterations, native resolution (1920×1080, 1653 frames),
$\lambda_{\text{dssim}}=0.2$, pose correction LR $2\times10^{-3}\to2\times10^{-4}$
(`pose_correction_method.md`), 24GB VRAM cap.

**Learning rates** (decay to 1/3 of init; position follows stock 3DGS's own schedule):

| parameter | initial | final |
|---|---|---|
| `position_lr` | 0.00016 | 0.0000016 |
| `scaling_lr` | 0.005 | 0.00167 |
| `rotation_lr` | 0.001 | 0.00033 |
| `opacity_lr` | 0.05 | 0.0167 |
| `feature_lr` | 0.0025 | 0.00083 |

**Densification/pruning**: full-duration window (`densify_until_iter =
iterations`), `densify_from_iter=500`, interval 100, grad threshold
$8\times10^{-5}$, opacity floor 0.0045. Periodic opacity reset disabled
(causes "crystal" artifacts on this scene, `feedback_opacity_reset_disabled.md`);
size-pruning kept independent of that via `size_prune_from_iter=3000`.

**Loss terms**: depth supervision on (weight $1.0\to0.01$), anchor loss on
(weight 1.0, constant). Anisotropy loss and density-based extent, both
described below, disabled.

**Result**: loss 0.088, L1 0.065, SSIM 0.834, 47,705 Gaussians, max_scale 0.293.

## 3. Reliability of evaluation metrics

PSNR disagreed with visual inspection twice, favoring bundle adjustment
(worse in practice) and penalizing pose correction (better in practice).
Tracks blur/sharpness, not multi-view consistency. Dropped as a metric;
not computed for most runs here.

Of the metrics still logged, none is fully reliable alone. Per
`docs/experiment_log.md`'s N1-N20 audit: SSIM tracks visual quality best
but still missed twice (favored a config the reviewer didn't prefer;
implied a gap between two configs judged equally good). Raw L1/loss is
least reliable, diverging 25%+ between visually-tied results. Gaussian
count and max_scale predict specific failure modes (sparsity,
over-pruning) more reliably than either loss metric. Treat all of these as
directional; confirm with a render before accepting or rejecting a change.

## 4. Method descriptions

**Pose correction** (`pose_correction_method.md`): learnable 6-DOF
correction per training camera, optimized jointly under the photometric
loss. Fixes inter-frame pose disagreement from LoGeR's independent
per-window inference. Most consequential mechanism in the project.

**Depth supervision** (`depth_supervision_method.md`): penalizes rendered
depth against a depth map reconstructed from LoGeR's own points, along the
sampled camera's ray. Derived from the same data as the point cloud, not
independent — weak alone, valuable combined with anchor loss.

**Anchor loss** (`anchor_loss_method.md`), the main new contribution:
penalizes squared displacement of each Gaussian from its point-cloud birth
position, tracked through clone/split so descendants inherit the same
anchor. View-independent, constrains all three spatial dimensions, unlike
depth supervision's single-ray constraint. Neither term alone matches the
combination.

Two mechanisms were built and evaluated but are off in N5:

**Anisotropy loss** (`anisotropy_loss_method.md`): hinge penalty on
Gaussians whose largest scale axis dominates the second-largest, curbing
needle artifacts. Works as intended but costs sharpness — not worth it once
the full densify window already bounded max scale on its own.

**Density-based extent** (`point_density_extent_method.md`): swaps
camera-baseline spread for the point cloud's local density (median
nearest-neighbor distance) as the densify/prune scale reference. Over-pruned
badly on its only trial; discarded without a recalibration attempt.
