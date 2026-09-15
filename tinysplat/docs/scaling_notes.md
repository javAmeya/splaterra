# Scaling tinysplat to large scenes — findings, fixes, and this week's plan

Context: `tinysplat` trains vanilla-style 3DGS (via `gsplat`) on point clouds + poses
produced by LoGeR. On a small object (toy house, plain COLMAP input) it works well.
On a large scene, iteration-0 (the point-cloud initialization) looked better than
anything training produced afterward — quality monotonically degraded as training
proceeded. This doc covers: what's actually going on, what was wrong in the code,
what the broader 3DGS community has hit and fixed for this exact class of problem,
and this week's plan.

## 0. ROOT CAUSE FOUND AND FIXED (2026-09-15, live on a RunPod A100 against the real
`inference.mov` scene, LoGeR ws=64/os=12/reset_every=0 → 400k-point COLMAP init →
tinysplat)

**The training/eval mismatch in the exposure-compensation module was the dominant
cause.** `train.py` rendered with `use_trained_exp=True` (a learned full 3x4 affine
color correction, one per training image) while every eval call — and the exported
`.ply` — used `use_trained_exp=False`, and `GaussianModel.capture()` never saves the
exposure parameters at all. The optimizer was free to minimize its *actual* training
loss by dumping appearance/geometry errors into each image's private exposure
correction instead of fixing the underlying Gaussians. Measured directly on this
scene (instrumented `train.py` to log both): L1 **with** exposure fell smoothly
0.283→0.079 over 30k iterations (the optimizer genuinely was minimizing its loss the
whole time) while L1 **without** exposure — what eval, PSNR, and the shipped `.ply`
all actually show — improved only until ~iter 4000, then reversed and finished worse
than it started (0.164→0.304). None of the "with exposure" fit is recoverable after
the fact, and it's not usable at deployment anyway: a novel view in a walkable game
env has no per-training-image correction to apply.

This explains why the toy house was fine: dense orbital multi-view coverage gives
the exposure module little room to hide real errors, since many views constrain the
same surface point. The big scene is a long-walkthrough capture — each surface point
is seen from a narrow range of angles — so the exposure module became an escape
hatch instead of a minor color correction.

**Fix**: train with `use_trained_exp=False`, identically to eval (`train.py`, the
single `render()` call in the training loop). Before/after on the exact same scene,
config, and 30k iterations:

| | iter 3000 | iter 4000 (old peak) | iter 15000 | iter 30000 |
|---|---|---|---|---|
| before fix | PSNR 11.9 | PSNR 14.7 | PSNR 10.0 | PSNR 9.7 |
| **after fix** | PSNR 17.7 | PSNR 17.8 | **PSNR 18.9** | **PSNR 19.3** |

PSNR is now monotonically increasing for the entire run. This is the single highest-
leverage fix found this week — do this first on any scene before touching anything
else below.

Two other things were checked and fixed/ruled out along the way (kept in §2 for the
record, since they're real, just not the dominant cause here):
- **Fixed, real bug, not dominant**: `train.py`'s oversized-Gaussian pruning
  (`size_threshold`) was accidentally gated on `opacity_reset_interval`, so pushing
  that interval past `iterations` to disable resets (§2b) *also* silently disabled
  size-based pruning. Fixed by decoupling into `opt.size_prune_from_iter`. Confirmed
  via checkpoint inspection: `max_scale` blew up 23x (0.73→17.0) without the fix,
  stayed controlled (0.73→1.6) with it — but PSNR degraded on nearly the same
  trajectory either way, proving this wasn't what was destroying quality.
- **Ruled out entirely**: densification aggressiveness (§2c). A probe run with
  `densify_until_iter=0` (clone/split/prune fully disabled, just the raw 400k init
  optimized) produced the *same* degradation curve as the normal run. Also ruled out:
  `cameras_extent` miscalibration (§2e) and the znear/zfar fix (§2a) — both were
  checked directly against this scene's actual point-cloud/camera geometry and
  turned out to be reasonably calibrated already, so neither was biting here. (The
  znear/zfar fix itself is still correct/worth keeping — it just wasn't the story on
  this particular scene.)

## 1. Why "iteration 0 is the best" is a real, well-known symptom

*(§0 above has the confirmed, data-backed mechanism for this specific pipeline. This
section is general background on why the symptom is plausible/known in the wider
literature — worth keeping in mind for scenes where §0's specific fix isn't enough.)*

3DGS quality is not guaranteed to be monotonically improving. Photometric loss
optimizes what the *available views* can see and constrain; anything under-constrained
(sparse coverage, texture-less regions, geometry the camera path skims but doesn't
orbit) is free to drift. In a small, densely-covered scene, most of the scene is
well-constrained, so this drift is invisible. In a large scene reconstructed from a
long video trajectory (LoGeR's use case), most surfaces are seen from only a few
nearby frames — so there's very little to stop the optimizer from moving Gaussians
away from the (good) initialization to chase local photometric gradients, and nothing
in vanilla adaptive density control (ADC) reliably removes the floaters this produces.
This is documented repeatedly in the literature as a known 3DGS failure mode for
large/sparse-view scenes, not something unique to this codebase — see §3. The good
news: it's also a well-studied problem with known mitigations.

## 2. Bugs / misconfigurations found in tinysplat itself

Ranked by how likely each is to be *the* dominant cause of "gets worse every iteration"
in a large scene specifically (vs. being invisible on the toy house).

### 2a. [FIXED] Hardcoded `znear=0.01` / `zfar=100.0` regardless of scene scale — likely primary cause
`tinysplat/tinysplat/camera.py:5` defaulted every camera to `znear=0.01, zfar=100.0`,
and nothing in `colmap_loader.py` ever overrode it. `gsplat.rasterization()` (called
from `renderer.py:4`) hard-culls any Gaussian whose projected depth falls outside
`[near_plane, far_plane]` — **from both rendering and densification gradient
accumulation** (gsplat's own default for `far_plane` is `1e10`, i.e. effectively
"don't cull," which is a strong hint 100.0 was never meant to be a real limit).

LoGeR/DUSt3R-style pointmap reconstruction has **no fixed physical scale** —
`loger/utils/geometry.py` even has a `robust_scale_estimation()` helper, confirming
the output scale is arbitrary/per-scene, not metric. So whatever unit system a given
LoGeR run happens to produce, there's no reason it fits inside a hardcoded 100-unit
far plane. On the toy house (small, dense COLMAP capture) this default was probably
never binding. On a genuinely "large scene," it's very plausible that a meaningful
fraction of the point cloud sits beyond `zfar` and is invisible to the renderer from
the start — those Gaussians get initialized from the (good) point cloud at iter 0,
then never receive gradient again because they're outside the frustum bound, while
densification keeps spending budget on the geometry that *is* inside the bound. The
net visual effect over training: near geometry gets increasingly overfit/floatery
(see 2b/2c below) while far geometry silently stops being represented — which reads
exactly as "it was better before training started."

**Fix applied**: `colmap_loader.py` now computes per-scene `znear`/`zfar` from the
actual depth distribution of the point cloud as seen by its own cameras (0.5th /
99.5th percentile of depth, not raw min/max, so a few noisy points can't corrupt the
range), and assigns it to every camera. Watch `znear`/`zfar` logged to wandb config
on your next run — if they come out wildly different from `0.01`/`100.0`, that
confirms this was live.

### 2b. [INTENTIONAL — confirmed, not a bug] `opacity_reset_interval=40000 > iterations=30000` disables periodic opacity reset
Correction from an earlier pass of this doc: I initially flagged this as a likely
typo and reverted it to upstream's `3000`. It's not a typo — periodic opacity reset
was tried and empirically produced bright, needle-like "crystal" artifacts scattered
around the splat, so it was deliberately disabled by pushing the interval past
`iterations`. **Reverted back to 40000 (disabled).**

This is worth understanding rather than treating as settled, because it's a real,
documented tradeoff, not a dead end:
- Periodic opacity reset works by forcing *every* Gaussian's opacity down to ~0.01
  simultaneously, so gradients can flow again through regions a high-opacity floater
  was blocking. But this is described in the literature as introducing "a small shock
  in the training trajectory" — resetting opacity is "particularly harmful for
  error-based densification methods, as it leads to misleading error statistics right
  after the hard reset, potentially triggering wrong densification decisions," and
  can cause the photometric loss to become strongly non-monotonic post-reset.
  [Revising Densification in Gaussian Splatting](https://arxiv.org/pdf/2404.06109)
- Separately, "needle" artifacts — sharp, spike-like Gaussians that go highly
  anisotropic to overfit sparse views, most visible from close-up or off-training
  viewpoints — are a named, independent 3DGS failure mode for undersampled/sparse
  coverage. A hard opacity reset immediately followed by a densification pass on a
  large, sparsely-covered scene is close to a worst case for triggering this: a huge
  batch of near-zero-opacity Gaussians all regain opacity from the same few nearby
  views at once, with no cross-view consistency check, right as densify_and_prune is
  deciding what to split. That combination plausibly *is* the "white crystals."
  [Needle-artifact discussion](https://arxiv.org/html/2406.04251v3)

So disabling reset entirely correctly avoided the crystals, but it also means
`opacity_cull` (2c below) is now the **only** floater-removal mechanism running for
the whole 30k iterations — which raises the stakes on getting 2c right, and makes it
worth trying a gentler alternative to a flat on/off toggle rather than settling for
"resets are off, permanently":
- **Gradual opacity decay** instead of a hard reset: nudge every Gaussian's opacity
  down by a fixed amount each densification interval (rather than slamming it to
  0.01 all at once), so genuinely-needed Gaussians drift back up smoothly instead of
  every Gaussian in the scene getting reset in lockstep. This targets the same
  floater-cleanup goal without the reset "shock." Proposed directly in the paper
  above as an alternative to the hard reset.
- **A single early reset only** (e.g. once at iteration 1500, none after) — gets one
  floater-cleanup pass in before most of the scene's Gaussians exist yet (so the
  "shock" hits a much smaller population), without the repeated destabilization of
  doing it every 3000 iterations for the whole run.
- 3DGS-MCMC's relocate/noise-based approach (§3.1) sidesteps hard resets entirely by
  making pruning/relocation part of a continuous SGLD update rather than a periodic
  shock — the principled version of "get the cleanup benefit without the reset spike."

None of these are applied — they're candidates for Day 2/4 of the plan below, to try
*if* floaters are still a problem with resets off and 2c retuned.

### 2c. [FLAGGED, not changed] Densification is tuned *more aggressive* than upstream, in exactly the direction that produces floaters on sparse-coverage scenes
```
densify_grad_threshold = 0.00008   # upstream default: 0.0002  (2.5x more permissive)
opacity_cull            = 0.002    # upstream default: 0.005  (2.5x more permissive)
```
Both were moved in the same direction: split/clone more eagerly, and cull less
aggressively. On a small, well-covered scene this mostly just means slightly more
Gaussians. On a large scene where the point cloud is sparse relative to scene extent
(exactly what you describe), the same gradient threshold gets crossed by many more
"real but under-constrained" viewspace gradients, producing many more Gaussians per
densification step — and with fewer views per surface to keep them honest, more of
those survive as floaters. This is a documented interaction, not a hypothesis: the
"gradient collision" issue AbsGS identifies (large under-covered Gaussians can't
split because per-pixel gradients partially cancel) gets *worse*, not better, the
lower you push the threshold, because you're now also splitting on noise.

This is left as a **variable to A/B this week** rather than "fixed," since it's
plausible whoever set it did so deliberately for the toy-house scene. First
experiment in the plan below: revert to upstream defaults for the large-scene run and
see if floater growth (now visible via the new `gaussians/*` wandb metrics) slows.

### 2d. [Feature gap, not a bug] Depth supervision is scaffolded but dead
`train.py` has a whole depth-loss branch (`viewpoint_cam.depth_reliable`,
`depth_mask`, `invdepthmap`) and `Camera.__init__` (`camera.py:5`) defaults
`depth_reliable=False`. `colmap_loader.py` never sets these fields on any camera it
builds, so **the depth loss never fires for any COLMAP-sourced run** — the point
cloud's geometric information is only ever used once, at `create_from_pcd`
initialization, and is never used again as an ongoing constraint.

This matters more than it would in a plain-COLMAP pipeline, because **you already
have per-frame depth from LoGeR** (that's the whole point of a pointmap model) — it's
sitting unused. Feeding LoGeR's own per-view depth back in as `invdepthmap`/`depth_mask`/
`depth_reliable=True` for each training camera would give the optimizer an ongoing
anchor against drifting away from the good initialization in exactly the
under-constrained regions where photometric loss alone is the weakest. This is the
single most-corroborated fix in the literature for "good initialization, degrades
during optimization on sparse/large scenes" (see §3.3). It's a feature to build, not
a one-line fix — planned for Day 2–3 below.

### 2e. [Note] `cameras_extent` is camera-spread only, not point-cloud extent
`scene.py:7`'s `get_scene_extent` (matches upstream 3DGS's `getNerfppNorm`) measures
the spread of *camera centers*, not the point cloud. For an orbit-style capture (the
toy house) camera spread tracks scene size reasonably well. For a long walkthrough
trajectory (LoGeR's actual use case) the camera path can be much smaller (a hallway)
or much larger (a building loop) than the geometry actually being reconstructed, and
`cameras_extent` feeds directly into the absolute densify/prune size thresholds
(`0.01 * extent`, `0.1 * extent`) and the position-LR spatial scale. If Day 1's
znear/zfar fix + reverted densify thresholds don't fully resolve things, this is the
next thing to instrument (log point-cloud bbox extent alongside camera extent, in
wandb, and compare).

## 3. What the broader 3DGS community has already run into (with fixes)

### 3.1 Floaters from vanilla adaptive density control (ADC)
- The original 3DGS heuristic (opacity-only pruning + periodic reset) is explicitly
  called out in follow-up work as *"failing to eliminate floaters"* in many scenes —
  it "requires multiple hyperparameters to be carefully tuned and can fail." Large
  scenes with sparse coverage are exactly where it fails most.
  [3D Gaussian Splatting as Markov Chain Monte Carlo](https://arxiv.org/html/2404.09591) (NeurIPS 2024)
- **AbsGS** identifies *gradient collision*: large, under-reconstructed Gaussians
  accumulate per-pixel gradients that partially cancel in direction, so their
  gradient *magnitude* never crosses the split threshold even though they badly need
  to split. Fix: use the homodirectional (absolute-value) gradient sum instead of the
  vector-sum gradient as the densify criterion.
  [AbsGS: Recovering Fine Details for 3D Gaussian Splatting](https://arxiv.org/pdf/2404.10484)
- **3DGS-MCMC** reframes densify/prune entirely as sampling from a probability
  distribution (SGLD updates + deterministic relocate/add steps) instead of hand-tuned
  clone/split/reset heuristics. Reported to depend much less on point-cloud
  initialization quality and to give precise, stable control over Gaussian count —
  worth trying as a drop-in ADC replacement if tuning upstream's heuristics doesn't
  converge this week. [3D Gaussian Splatting as Markov Chain Monte Carlo](https://arxiv.org/html/2404.09591)
- **StableGS** traces persistent floaters specifically to *vanishing gradients* during
  backprop for certain floater configurations (not just an opacity/pruning issue),
  and proposes cross-view depth consistency + a dual-opacity model.
  [StableGS](https://arxiv.org/pdf/2503.18458)

### 3.2 Sparse SfM/point-cloud initialization in large or texture-poor scenes
- Large scenes with texture-less surfaces get poor SfM point coverage; 3DGS then has
  "difficult optimization and low-quality renderings" because ADC has to invent
  geometry in those regions from photometric gradient alone, with weak constraints —
  directly matches "the point cloud is insanely good quality, but sparse."
  [Springer review, 3DGS for sparse-view reconstruction](https://link.springer.com/article/10.1007/s10462-025-11171-4)
- Multiple recent works (Pi-GS, LM-Gaussian, D²GS, DepthRegGS) converge on the same
  fix: use a *dense* geometric prior (a feed-forward pointmap/depth model — which
  LoGeR already is) both for initialization *and* as ongoing per-view depth
  regularization during optimization, not just at t=0.
  [Pi-GS: Sparse-View GS with Dense π³ Initialization](https://arxiv.org/pdf/2602.03327),
  [D²GS depth-and-density guided dropout](https://arxiv.org/html/2604.05715v1)

### 3.3 Depth supervision as the anchor against drift
- "Enhanced 3DGS via Depth Priors, Adaptive Densification, and Denoising" integrates
  metric depth (Depth-Anything V2) as a dynamically-weighted regularization term
  specifically to keep optimization from drifting away from a good geometric prior,
  plus floating-artifact detection tied to depth disagreement.
  [PMC12656154](https://pmc.ncbi.nlm.nih.gov/articles/PMC12656154/)
- Caveat consistently raised: naively trusting a monocular/feed-forward depth prior
  everywhere can hurt, because such priors have local inaccuracies and
  cross-view inconsistency — the fix is to *mask* depth supervision to
  confident/consistent regions rather than applying it uniformly.
  [In Depth We Trust](https://arxiv.org/html/2604.05715v1)
  This directly informs how to wire up LoGeR's depth in §2d — don't blindly trust
  every pixel; use LoGeR's own confidence output (if it has one) or a cross-frame
  consistency check as the `depth_mask`.

### 3.4 Unbounded / wide depth-range scenes
- Explicit 3DGS-style methods "face challenges describing distant points in
  unbounded Euclidean space" — several works apply NeRF++-style space contraction or
  a two-layer (near/far) representation instead of one flat near/far frustum.
  [HoGS: Unified Near and Far Object Reconstruction](https://arxiv.org/pdf/2503.19232)
  Relevant if, after the znear/zfar fix, you still have scenes with an extreme
  near-to-far ratio (e.g., a close wall a few units away and a horizon hundreds of
  units away in the same shot) — depth-buffer precision degrades badly at high
  ratios regardless of the absolute near/far values chosen.

### 3.5 Scaling to genuinely large scenes on constrained hardware
Directly relevant to the RTX 4090 constraint and the "walkable game env" end goal —
these are architectural strategies, not bug fixes, for when a single, flat 3DGS
model stops fitting in memory/compute at all:
- **VastGaussian**: partitions a large scene into overlapping cells, trains each with
  its own appearance encoding, merges — explicitly designed to keep training within a
  single GPU's memory budget on large aerial/drone captures, and reports the
  partitioning itself "mitigates floater occurrence" as a side effect (each cell has
  a much better view-to-geometry ratio than the whole scene at once).
  [VastGaussian](https://arxiv.org/pdf/2402.17427)
- **CityGaussian / CityGaussianV2**: divide-and-conquer training + Level-of-Detail
  (LoD) rendering, so distant content uses fewer/coarser Gaussians — good match for a
  "walkable env" where the player is only ever near a fraction of the full scene.
  [CityGaussianV2](https://arxiv.org/pdf/2411.00771)
- **Hierarchical 3D Gaussian Splatting**: builds an explicit LOD hierarchy with
  smooth level transitions for real-time rendering of very large scenes — most
  directly aligned with your stated end goal (walkable game environment), since it's
  optimized for interactive traversal, not just offline reconstruction quality.

None of these are "implement this week" scope, but they're the right next step
*after* the current single-chunk pipeline is correctness-verified on a large scene —
see Day 5 below.

## 4. How to tell which fix actually mattered (don't just eyeball renders)

Use wandb (already wired up) as the diagnostic instrument, not just a render viewer.
New metrics added this session (`train.py`, logged every 10 iters):

| metric | what it tells you |
|---|---|
| `gaussians/mean_scale`, `gaussians/max_scale` | Gaussians ballooning in size = floaters or bad splits. Should stay bounded, not trend upward for the whole run. |
| `gaussians/mean_opacity` | A slow creep toward the reset floor right after each `opacity_reset_interval` tick, recovering afterward, is healthy. Monotonic drift with no reset-driven dips means resets still aren't taking effect. |
| `gaussians/frac_low_opacity` | Spikes right after densification = lots of newly-created Gaussians that immediately fail to earn their keep — a sign the split/clone threshold is too permissive for this scene. |
| `gaussians/frac_visible_this_view` | If this stays low relative to `train/num_gaussians` for most of the run, you're spending Gaussian budget on things most views never see/correct — a scene-partitioning signal (§3.5), not a hyperparameter one. |
| `train/ssim` vs `train/num_gaussians` | Overlay these. If SSIM peaks then declines *while* Gaussian count keeps climbing, densification is actively hurting past that point — a strong argument for reverting 2c or trying MCMC-style ADC (§3.1). |
| test PSNR vs train PSNR gap (already computed at `testing_iterations`) | A growing gap = overfitting to the training views specifically, consistent with floaters that look fine from training viewpoints but wrong from novel ones. |

**Rule for this week: change exactly one variable per run**, and always compare
against the same fixed eval cameras/checkpoint-iteration list already in `train.py`
(`testing_iterations`). Don't stack the 2a fix and a 2c/2b experiment in one run and
try to attribute the result — you already have 2a applied as of this session (2b is
confirmed intentional and left as-is); the first big-scene run this week is the
baseline to re-measure against.

## 5. This week's plan

**Day 1 — DONE.** Root cause found and fixed live against the real scene on a RunPod
A100 (§0): the `use_trained_exp` train/eval mismatch. PSNR now climbs monotonically
17.7→19.3 over 30k iterations instead of degrading 14.7→9.7. znear/zfar (2a) and
`size_prune_from_iter` (new, in §2b) fixes are applied too; densification
aggressiveness (2c) and `cameras_extent` (2e) were checked and ruled out as
contributors *on this scene* via a no-densify probe run and direct geometry
inspection — see §0 for the receipts. Artifacts from every run (`.ply` exports +
full logs, baseline vs. each fix in isolation) are in `pod_results/` for
comparison/viewing.

**Day 2 — Confirm the exposure fix generalizes, then decide if densification/reset
tuning is still worth it**
- Re-run on a second scene (different capture, ideally a shorter one to iterate
  fast) with the exposure fix to make sure the 17→19 PSNR climb wasn't specific to
  `inference.mov`. If it holds, this fix is the new baseline for every future run —
  stop re-deriving it.
- Now that the dominant bug is gone, re-evaluate whether densification aggressiveness
  (2c: `0.00008`/`0.002` vs. upstream `0.0002`/`0.005`) and the opacity-reset
  question (2b) still matter now that the fixed run's own trajectory is available as
  a clean baseline to compare against — they were ruled out as *dominant* causes, not
  necessarily proven irrelevant in an otherwise-healthy run. A/B them against §4's
  `gaussians/*` diagnostics the same way as before.

Since periodic hard opacity reset is off (2b) and won't be turned back on wholesale
(it produces the crystal artifacts), Day 2 is also a good point to try one of the
gentler alternatives from §2b: gradual opacity decay, or a single early reset at
~iteration 1500 only. Compare against "resets fully off" on `gaussians/frac_low_opacity`
and visual floater count.

**Day 3 — Wire up LoGeR depth as ongoing supervision (2d)**
- Extract LoGeR's per-frame depth/pointmap output for the same frames you already
  have COLMAP poses for; populate `depth_reliable=True`, `invdepthmap`, `depth_mask`
  on each training `Camera`. Start with `depth_mask` = "high-confidence" pixels only
  (LoGeR's own confidence if it outputs one, or a simple reprojection-consistency
  check across neighboring frames) — do not trust it uniformly (§3.3 caveat).
- Compare against Day 2's winner with depth supervision on vs. off, same scene, same
  seed. This is the fix most likely to specifically close the "iter-0 was better"
  gap, since it directly re-anchors optimization to the point cloud you already
  trust.

**Day 4 — Consider 3DGS-MCMC-style ADC if heuristic tuning plateaus**
- Only if Days 1–3 still show floater accumulation despite correct near/far, correct
  reset cadence, tuned thresholds, and depth anchoring: swap the clone/split/reset
  heuristics for MCMC-style relocate/add updates (§3.1). This is a bigger code
  change (rewrites `densify_and_prune` and the training-loop noise injection) — worth
  it only if the heuristic path is demonstrably stuck, since it changes the
  optimization dynamics more broadly.

**Day 5 — Full-scene validation + plan the LOD/partitioning step**
- Run the best config from Days 1–4 on the complete large scene, full 30k iterations.
- Whatever the 4090 memory ceiling turns out to be (log peak VRAM + final Gaussian
  count in wandb), that number is your hard constraint for scaling further. If it's
  close to full, this is the point to scope VastGaussian/CityGaussian-style
  partitioning or a distance-based LOD (§3.5) — needed regardless for a walkable
  game env, since real-time rendering of one flat, full-detail large-scene 3DGS on a
  single 4090 is a separate problem from training it.

## Sources

- [3D Gaussian Splatting as Markov Chain Monte Carlo](https://arxiv.org/html/2404.09591)
- [AbsGS: Recovering Fine Details for 3D Gaussian Splatting](https://arxiv.org/pdf/2404.10484)
- [StableGS: A Floater-Free Framework for 3D Gaussian Splatting](https://arxiv.org/pdf/2503.18458)
- [A review on 3D Gaussian splatting for sparse view reconstruction (Springer)](https://link.springer.com/article/10.1007/s10462-025-11171-4)
- [Pi-GS: Sparse-View Gaussian Splatting with Dense π³ Initialization](https://arxiv.org/pdf/2602.03327)
- [In Depth We Trust: Reliable Monocular Depth Supervision for Gaussian Splatting](https://arxiv.org/html/2604.05715v1)
- [Enhanced 3D Gaussian Splatting for Real-Scene Reconstruction via Depth Priors, Adaptive Densification, and Denoising (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12656154/)
- [HoGS: Unified Near and Far Object Reconstruction via Homogeneous Gaussian Splatting](https://arxiv.org/pdf/2503.19232)
- [VastGaussian: Vast 3D Gaussians for Large Scene Reconstruction](https://arxiv.org/pdf/2402.17427)
- [CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes](https://arxiv.org/pdf/2411.00771)
- [gsplat rasterization API docs (near_plane/far_plane/radius_clip defaults)](https://docs.gsplat.studio/main/apis/rasterization.html)
