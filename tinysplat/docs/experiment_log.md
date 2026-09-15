# tinysplat experiment log

Terse, per-run reference: exact config, what changed vs. the previous run, result,
verdict. For the narrative writeup (root-cause analysis, literature survey,
day-by-day plan) see `docs/scaling_notes.md` — this file is the lab notebook.

All runs use `inference.mov` (1653 frames, 1920x1080 @ 30fps) unless noted.
LoGeR settings are the same for every run below (`--window_size 64
--overlap_size 12 --reset_every 0`, checkpoint=`LoGeR`) — that part never
changed this session, so it's not repeated per-entry.

## Confirmed good, 2026-09-15 (user's own visual inspection, not just PSNR)

Three results from today are confirmed good and worth building on:

1. **Run 5 — exposure fix (`FIXED_noexpmismatch`)**: the single highest-leverage
   fix. Train with `use_trained_exp=False`, matching eval — apply this on any
   scene before anything else below.
2. **Run 9 — pose correction v1** (`PoseCorrection`, constant `lr=5e-4`):
   structurally much better than every non-pose-correction run — parallax
   duplicates nearly eliminated. PSNR is a **misleading** metric for this
   comparison (v1 scores *below* Run 6 numerically despite looking clearly
   better) — it appears to track blur/sharpness much more than multi-view
   structural consistency, which is what actually matters for a walkable scene.
3. **Run 10 — pose correction v2** (decaying `lr: 2e-3 → 2e-4`): also confirmed
   good by the user despite scoring *below* v1 on PSNR (15.59 vs 16.81) — same
   "PSNR misleads here" pattern. Comparative structural quality vs. v1 not yet
   explicitly ranked by the user; both are validated as good outcomes.

**Practical implication**: stop using PSNR as the primary decision metric once
pose correction is in play. Always pull the `.ply` and get a free-navigation
look before calling a change a win or a loss — this is the second time this
session PSNR pointed the wrong way (see Run 8's revert).

**Recommended config as of today**: exposure fix (Run 5) + pose correction
(Run 9 or 10 — pick based on further visual comparison) + shared focal length
(Run 6, marginal but correct) + `size_prune_from_iter` fix (Run 2). Depth
supervision (Run 7) and COLMAP bundle adjustment (Run 8) are NOT part of the
recommended config — see their entries below for why.

---

## Run 1 — baseline
**Date:** 2026-09-15
**Converter:** per-frame independently-fit focal length, `conf_threshold=20%`, `max_points=400000`, no frame filtering, no depth export.
**Training config:** `opacity_reset_interval=40000` (disabled, intentional — see feedback memory), oversized-Gaussian pruning still buggy (`size_threshold` gated on `opacity_reset_interval`, so it never fires either), `densify_grad_threshold=0.00008`, `opacity_cull=0.002`, `use_trained_exp=True` (bug, not yet found), no depth supervision, `dataset.eval=True` (llffhold=8), 30k iterations.
**Result:** PSNR 11.88 → 14.63 (peak @ iter 4000) → 9.77 @ iter 30000. `max_scale` blew up 23x (0.73→17.0) over the run.
**Verdict:** Reproduces the original "iteration 0 was best, degrades after" complaint. `.ply`/log: `pod_results/baseline_iter30000.ply`.

---

## Run 2 — size-prune fix
**Date:** 2026-09-15
**Change from Run 1:** Added `opt.size_prune_from_iter=3000`, decoupling oversized-Gaussian pruning from `opacity_reset_interval` (which is deliberately disabled). Nothing else changed.
**Result:** PSNR 11.88 → 14.66 → 9.73 (nearly identical to Run 1). `max_scale` now controlled (0.73→1.6 instead of →17.0).
**Verdict:** Real bug, correctly fixed, but *not* the dominant cause of the degradation — trajectory barely moved despite `max_scale` being fixed. `.ply`: `pod_results/sizeprune_fixed_iter30000.ply`.

---

## Run 3 — no-densification probe
**Date:** 2026-09-15
**Change from Run 2:** `densify_until_iter=0` (clone/split/prune/opacity-reset entirely disabled — just the raw 400k-point init optimized directly). Diagnostic only, not a candidate config.
**Result:** PSNR 11.39 → 14.13 → 9.59. Same degradation shape as Runs 1–2.
**Verdict:** Rules out densification aggressiveness as the cause — degradation happens even with zero densification. `.ply`: `pod_results/nodensify_probe_iter30000.ply`.

---

## Run 4 — exposure-mismatch diagnostic
**Date:** 2026-09-15
**Change from Run 2:** Instrumented `train.py` to additionally render a train-camera sample with `use_trained_exp=True` (matching what training actually optimizes) vs `False` (what eval/`.ply` show) at every eval checkpoint, logging both L1 values.
**Result:** PSNR 11.13 → 14.26 → 9.68 (same shape). L1 **with** exposure fell smoothly 0.283→0.079 the whole run (optimizer genuinely minimizing its loss); L1 **without** exposure improved only to iter ~4000 then reversed, finishing worse than it started (0.164→0.304).
**Verdict:** Confirms the root cause — training and eval were optimizing/measuring two different things, and the gap between them only grows. `.ply`: `pod_results/diag_iter30000.ply`.

---

## Run 5 — exposure fix ⭐ (`FIXED_noexpmismatch`)
**Date:** 2026-09-15
**Change from Run 4:** `train.py`'s render call switched from `use_trained_exp=True` to `use_trained_exp=False` — training now optimizes the exact same thing eval measures.
**Result:** PSNR 17.67 → 17.82 → 18.29 → 18.88 → 19.19 → **19.30**, monotonically increasing for the entire run. First healthy trajectory of the session.
**Verdict:** The single highest-leverage fix found this week. `.ply`: `pod_results/FIXED_noexpmismatch_iter30000.ply`.

---

## Run 6 — shared focal length
**Date:** 2026-09-15
**Change from Run 5:** Converter (`loger_to_colmap.py`) changed from an independent per-frame focal-length fit to ONE shared focal length pooled across all frames (measured ~5% frame-to-frame jitter in the per-frame version — real single-lens video, so that jitter is noise, not signal). Result: `fx=286.75, fy=286.52` for the whole capture.
**Result:** PSNR 17.67 → 17.81 → 18.21 → 18.84 → 19.18 → **19.32** — essentially identical to Run 5 numerically.
**Verdict:** Correct fix, real (if marginal) visual improvement on some frames when actually rendered and compared — but not the dominant remaining issue. `.ply`: not separately exported (numerically ~same as Run 5); renders in `pod_results/vis/sharedfocal_*.png`.

---

## Run 7 — depth supervision (tried, not adopted)
**Date:** 2026-09-15
**Change from Run 6:** Wired up the depth-loss machinery that already existed in `train.py` but was never fed anything. `loger_to_colmap.py` now also exports per-frame `depths/*.npz` (invdepth + confidence mask, reconstructed from LoGeR's own camera-frame pointmap `local[...,2]`); `colmap_loader.py` loads them and sets `depth_reliable=True` per camera.
**Result:** PSNR 17.14 → 17.43 → 17.88 → 18.52 → 19.00 → 19.14 — slightly *worse* than Run 6. Visually: ghosting and the near-camera floater frame unchanged.
**Verdict:** Not useful, and the reason is structural, not just "needs tuning": the "depth" fed to supervision is reconstructed from the exact same LoGeR points/poses that already initialized the Gaussians — it's not an independent constraint, just a per-view re-anchoring to already-noisy single-frame data. Doesn't touch cross-frame disagreement, which is what actually causes the visible artifacts. Left wired up in the code (harmless dead weight if `depths/` doesn't exist) but not part of the recommended config. `.ply`: `pod_results/vis/depthsup_*.png` (renders only, no separate `.ply` exported).

---

## Run 8 — COLMAP bundle adjustment (reverted)
**Date:** 2026-09-15
**Change from Run 6:** New `tinysplat/colmap_refine.py` (via `pycolmap`, installed on pod — no `colmap` CLI needed): SIFT feature extraction → sequential matching (video-order) → triangulate 3D points from LoGeR's poses (held fixed) + verified 2D correspondences → global bundle adjustment (jointly refines poses *and* points). Output: 250,520 points with mean track length 7.5 (vs. LoGeR's raw points, each from exactly one frame).

Two real `pycolmap` 4.2.0 bugs hit and worked around along the way: `extract_features` assigns image_ids by some internal (not sorted-filename) order; `Reconstruction.transcribe_image_ids_to_database()` only remaps `Image.image_id`, not the corresponding `Frame.frame_id`/`data_ids`, desyncing them worse and tripping `triangulate_points`'s consistency check. Fix used: read the database's own id-per-filename assignment first, then write the known-pose model directly with those exact ids (bypassing `transcribe` entirely).

**Result:** PSNR 18.41 → 18.50 → 19.20 → 19.96 → 20.39 → **20.52** — highest PSNR of the session, never plateaus. My own fixed-test-camera render comparison: real, visible sharpening on well-textured hard surfaces (pavement patterns much crisper); corridor ghosting only marginally better; the near-camera floater frame completely unchanged.
**Verdict:** ⚠️ **REVERTED per the user's own visual inspection** — despite higher PSNR and better fixed-angle renders, free navigation of the actual `.ply` looked *worse* overall than Run 6. Take PSNR and fixed-angle renders as suggestive, never sufficient — always get a free-navigation look before calling a change a win. `dataset.source_path` reverted to the Run 6 dataset; the COLMAP-refined dataset and its checkpoints deleted from the pod. `colmap_refine.py` and `pycolmap` left in place (harmless, unused) in case worth revisiting later. `.ply`: `pod_results/COLMAP_REFINED_iter30000.ply` (kept for reference despite the revert).

---

## Run 9 — differentiable pose correction v1
**Date:** 2026-09-15
**Change from Run 6:** New `tinysplat/tinysplat/pose_correction.py` (`PoseCorrection` class) — every **training** camera (never test/held-out — see class docstring) gets a small learnable 6-DOF SE(3) delta (3 axis-angle rotation via Rodrigues' formula + 3 translation), zero-initialized, optimized by the *same* photometric loss as the Gaussians via its own Adam optimizer, constant `lr=5e-4` for the whole run. `renderer.py`'s `render()` gained an optional `view_matrix` override so the corrected pose flows through the same differentiable rasterization call.
**Result:** PSNR 16.58 → 16.85 → 16.91 (peak @ iter 7000) → 16.90 → 16.82 → **16.81** — numerically *worse* than Run 6, plateaus early.
**Verdict (user's own visual inspection, overriding the PSNR read):** "structurally much much better than anything else... parallax duplicates got almost zero... still a little blurry, still some shards but considerable improvement." PSNR is judged a poor proxy here — it's apparently more sensitive to blur/sharpness than to the multi-view structural consistency that actually matters for a walkable scene. `.ply`: `pod_results/POSE_CORRECTION_B_iter30000.ply`.

---

## Run 10 — pose correction v2, more aggressive
**Date:** 2026-09-15
**Change from Run 9:** `PoseCorrection` LR made a decaying schedule instead of constant — `lr_init=2e-3` (4x higher than v1) → `lr_final=2e-4`, exponential decay over the full 30k iterations (same philosophy as the Gaussian xyz LR schedule: move aggressively while the scene is still coarse, settle down late so fine-detail sharpening isn't chasing a pose still drifting underneath it). Also logs `pose_corr/mean_rotation_rad`, `pose_corr/mean_translation`, `lr/pose_corr` to wandb.
**Result:** PSNR 15.41 → 15.79 → 15.97 → **16.02 (peak @ iter 15000)** → 15.80 → **15.59** — worse than v1 (16.81) and shows a peak-then-decline shape v1 didn't. `.ply`: `pod_results/POSE_CORRECTION_B_v2_iter30000.ply`.
**Verdict:** Confirmed good by the user (2026-09-15), despite scoring below v1 on PSNR — consistent with the "PSNR misleads once pose correction is active" pattern established by v1. Not yet explicitly ranked against v1 for which is structurally better; both stand as validated, usable results. The peak-then-decline PSNR shape (absent in v1) remains a mechanistic caution sign worth revisiting if v2 is picked for further iteration (see the paper-style method writeup in the session transcript for a candidate explanation: no warmup delay before pose gradients are admitted, so the larger v2 LR may act on less-reliable early gradients).

---

## Open / not yet tried
- **Frame-level exclusion** for the near-camera floater artifact (traced to one video frame with 12,364 Gaussians collapsed within 0.3 units of that camera — likely motion blur or lens occlusion in the source footage). Confidence-mean filtering already in `loger_to_colmap.py` (`--min_conf_frac`) didn't catch it since that frame's *average* confidence wasn't low enough. Needs a stricter/different per-frame criterion.
- **Densification aggressiveness** (`densify_grad_threshold=0.00008` vs. upstream `0.0002`, `opacity_cull=0.002` vs. upstream `0.005`) — ruled out as the *dominant* cause (Run 3) but never re-tested against the current healthy baseline.
- Re-running COLMAP refinement (Run 8) *combined with* pose correction, now that pose correction alone is known to help structurally — untested combination.
