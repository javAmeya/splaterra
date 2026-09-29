# Depth Supervision from the Point-Cloud Prior

First tried in isolation and found weak; revived alongside anchor loss on
the expectation that constraining depth along each camera's ray would
complement anchor loss's full-3D constraint rather than duplicate it.

Implementation: `tinysplat/train.py` (loss computation), depth export in
`tinysplat/loger_to_colmap.py`, loading in `tinysplat/tinysplat/colmap_loader.py`.

## 1. Where the depth ground truth comes from

There is no independent depth sensor or external depth estimator in this
pipeline. The "ground truth" inverse depth map used here is **reconstructed
from LoGeR's own output** — the same per-frame camera-space pointmap
(`local[..., 2]`, the z-coordinate of each pixel's estimated 3D position in
camera space) that already seeded the Gaussian point cloud in the first
place. `loger_to_colmap.py` exports one `depths/*.npz` per frame containing
this reconstructed depth plus a per-pixel confidence mask
(`depth_reliable`/`depth_mask` downstream), rather than depth coming from
any source independent of the point cloud.

This is the load-bearing limitation of the whole mechanism: depth
supervision is **not an independent constraint** on the optimized scene —
it is a re-statement, along each camera's ray, of information the point
cloud already contributed at initialization. It cannot correct an error in
LoGeR's own estimate, only encourage the optimized Gaussians not to drift
away from it. (Contrast with `tinysplat/colmap_refine.py`'s SIFT-based
bundle adjustment, which establishes correspondences directly from image
content — see `docs/pose_correction_method.md` §4 for the same
independent-vs.-derived distinction in a different context.)

## 2. Formulation

For training camera $k$ with per-pixel rendered depth
$D_k \in \mathbb{R}^{H\times W}_{>0}$ (from the rasterizer's depth output,
`render_pkg["depth"]`) and LoGeR-derived per-pixel monocular inverse depth
$\hat{D}_k^{-1}$ (`viewpoint_cam.invdepthmap`, precomputed at export time)
with validity mask $M_k \in \{0,1\}^{H\times W}$
(`viewpoint_cam.depth_mask`), the loss compares **inverse** depths, not raw
depths — standard practice for monocular depth supervision since it
compresses the dynamic range of far-away geometry and avoids the loss being
dominated by a small number of very-far pixels:

$$D_k^{-1}(p) = \frac{1}{D_k(p) + \epsilon}, \quad p \in \{p : M_k(p) = 1\}$$

with $\epsilon = 10^{-6}$ for numerical safety. The loss for camera $k$ is
the masked mean absolute error:

$$\mathcal{L}_{\text{depth}}(k) = \frac{1}{\lvert\{p : M_k(p)=1\}\rvert} \sum_{p:\,M_k(p)=1} \left\lvert D_k^{-1}(p) - \hat{D}_k^{-1}(p) \right\rvert$$

added to the total loss with a **decaying** weight schedule
$w_{\text{depth}}(s)$ (`get_expon_lr_func`, exponential decay from
`depth_l1_weight_init` to `depth_l1_weight_final` over `opt.iterations`
steps) — in N5, $1.0 \to 0.01$:

$$\mathcal{L}_{\text{total}} \mathrel{+}= w_{\text{depth}}(s)\,\mathcal{L}_{\text{depth}}(k_s)$$

Depth supervision is only applied when **both** $w_{\text{depth}}(s) > 0$
and the sampled camera has `depth_reliable = True` (set per-frame at export
time based on confidence-mask coverage) — unlike anchor loss and anisotropy
loss, this term is view-conditional, contributing gradient only on
iterations where $k_s$ happens to be a reliable-depth frame.

## 3. A masked-division numerical fix (already in place)

Background/sky/very-far pixels can have $D_k(p) \approx 0$ in this
codebase's depth convention, which would make $1/(D_k(p)+\epsilon)$ blow up
to `inf`. Since `inf * 0` (from multiplying by the mask $M_k$, which should
zero out exactly these invalid pixels) evaluates to `nan` in IEEE
floating-point, not `0`, a naive `(pred_invdepth - gt) * mask` would poison
the entire loss/gradient with `nan` even though the masked pixels are
supposed to be excluded. The fix (`train.py`, "FIX #8"): substitute a safe
denominator (`1.0`) on masked-out pixels *before* the division, so `inf`
never enters the computation in the first place:

```python
safe_pred_depth = torch.where(depth_mask_bool, pred_depth, torch.ones_like(pred_depth))
pred_invdepth = 1.0 / (safe_pred_depth + 1e-6)
```

## 4. Empirical results

- **Alone** (Run 7, 2026-09-15, pre-pose-correction): slightly *worse*
  PSNR than the baseline (18.52→19.14 vs. 19.00 baseline trajectory),
  visually unchanged ghosting/floater. First evidence the mechanism is weak
  in isolation.
- **Alone, arm N2** (N1-N8 sweep): 60k ssim 0.674 — mid-pack, not a
  standout on its own.
- **Combined with anchor loss, N5** (D+A): ssim **0.834**, the best 60k
  result of the whole anchor-weight family. The combination clearly
  outperforms either alone (N2's 0.674 depth-only, N3's 0.622 anchor-only)
  — see `docs/anchor_loss_method.md` §4 for the complementarity argument
  (depth constrains along-ray, anchor constrains full 3D).

**Net conclusion**: depth supervision is weak-to-mediocre alone,
sometimes actively harmful, but synergizes with anchor loss in the N5
configuration. The mechanism should probably not be re-tested in isolation
again — its value is specifically in the combination.

## 5. Open directions

- **Confidence-weighted loss**: the export pipeline already computes a
  per-point confidence value (used for the boolean `depth_reliable`/
  `depth_mask` gating); the loss itself is currently unweighted within the
  valid region — a soft confidence weighting (rather than hard mask) is
  untried.
- **Why does combination help more than either term alone?** Not
  mechanistically explained yet, beyond the along-ray-vs-full-3D
  complementarity argument in §4 — worth a closer look at whether it's
  genuinely additive/complementary constraint coverage, or whether depth
  supervision is doing something more subtle like stabilizing early
  training before anchor loss's gradient becomes reliable.
