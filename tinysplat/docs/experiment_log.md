# tinysplat experiment log

Terse per-run reference: config change, result, verdict. All runs use `inference.mov` (1653 frames, 1920x1080@30fps) unless noted. LoGeR settings constant throughout (`--window_size 64 --overlap_size 12 --reset_every 0`, checkpoint=`LoGeR`).

## Confirmed good, 2026-09-15 (user's own visual inspection, not just PSNR)

1. **Run 5 — exposure fix** (`FIXED_noexpmismatch`): highest-leverage fix. Train with `use_trained_exp=False`, matching eval.
2. **Run 9 — pose correction v1** (`PoseCorrection`, const `lr=5e-4`): structurally much better, parallax duplicates nearly eliminated, despite scoring *below* Run 6 on PSNR (16.81 vs Run6's 19.32).
3. **Run 10 — pose correction v2** (decaying `lr: 2e-3→2e-4`): also confirmed good despite scoring below v1 on PSNR (15.59 vs 16.81) — same "PSNR misleads" pattern.

**Practical implication:** stop trusting PSNR as primary metric once pose correction is active — always inspect the `.ply` in free navigation.

**Recommended config as of 2026-09-15:** exposure fix (Run 5) + pose correction (Run 9 or 10) + shared focal length (Run 6) + `size_prune_from_iter` fix (Run 2). Depth supervision (Run 7) and COLMAP BA (Run 8) excluded.

---

## Run 1 — baseline
Converter: per-frame focal, `conf_threshold=20%`, `max_points=400000`. Config: `opacity_reset_interval=40000` (disabled, intentional), `densify_grad_threshold=0.00008`, `opacity_cull=0.002`, `use_trained_exp=True` (bug, not yet found), 30k iters, `eval=True` (llffhold=8).
**Result:** PSNR 11.88 → 14.63 (peak@4000) → 9.77@30000. `max_scale` blew up 23x (0.73→17.0).
**Verdict:** reproduces "iter0 best, degrades after." `.ply`: `RUN1_30k.ply`.

## Run 2 — size-prune fix
Added `opt.size_prune_from_iter=3000`.
**Result:** PSNR 11.88→14.66→9.73 (~same as Run1). `max_scale` controlled (0.73→1.6).
**Verdict:** real bug, correctly fixed, not the dominant degradation cause. `.ply`: `RUN2_30k.ply`.

## Run 3 — no-densification probe
`densify_until_iter=0` (diagnostic only).
**Result:** PSNR 11.39→14.13→9.59. Same degradation shape.
**Verdict:** rules out densification aggressiveness as the cause. `.ply`: `RUN3_30k.ply`.

## Run 4 — exposure-mismatch diagnostic
Logged L1 with/without `use_trained_exp` at every checkpoint.
**Result:** PSNR 11.13→14.26→9.68. L1 **with** exposure: 0.283→0.079 (smooth). L1 **without**: improved to iter~4000 then reversed, 0.164→0.304 (worse than start).
**Verdict:** confirms root cause — train/eval optimizing different targets. `.ply`: `RUN4_30k.ply`.

## Run 5 — exposure fix ⭐ (`FIXED_noexpmismatch`)
`use_trained_exp=True→False`.
**Result:** PSNR 17.67→17.82→18.29→18.88→19.19→**19.30**, monotonic. First healthy trajectory of the session.
**Verdict:** highest-leverage fix of the week. `.ply`: `RUN5_30k.ply`.

## Run 6 — shared focal length
Converter switched to one shared focal length: `fx=286.75, fy=286.52`.
**Result:** PSNR 17.67→17.81→18.21→18.84→19.18→**19.32** — ~identical to Run5.
**Verdict:** correct, real but marginal improvement; not the dominant issue. Renders: `sharedfocal_*.png`.

## Run 7 — depth supervision (tried, not adopted)
Wired existing depth-loss machinery; exported per-frame `depths/*.npz`.
**Result:** PSNR 17.14→17.43→17.88→18.52→19.00→19.14 — slightly worse than Run6. Ghosting/floater unchanged.
**Verdict:** not useful — depth is reconstructed from the same LoGeR points already used for init, not an independent constraint. Left wired but not in recommended config. Renders: `depthsup_*.png`.

## Run 8 — COLMAP bundle adjustment (reverted)
New `colmap_refine.py` (`pycolmap`): SIFT → sequential match → triangulate (poses fixed) → global BA. Output: 250,520 points, mean track length 7.5.
**Result:** PSNR 18.41→18.50→19.20→19.96→20.39→**20.52** — highest PSNR of the session. Fixed-camera renders: visibly sharper pavement texture; floater unchanged.
**Verdict:** ⚠️ **REVERTED** — free navigation of the `.ply` looked worse overall despite higher PSNR. Take PSNR/fixed-angle renders as suggestive only. `colmap_refine.py`/`pycolmap` left in place unused. `.ply`: `RUN8_30k.ply` (kept for reference).

## Run 9 — differentiable pose correction v1
New `pose_correction.py` (`PoseCorrection`): per-training-camera 6-DOF SE(3) delta, zero-init, own Adam, const `lr=5e-4`.
**Result:** PSNR 16.58→16.85→16.91(peak@7000)→16.90→16.82→**16.81** — numerically worse than Run6.
**Verdict (user, overriding PSNR):** "structurally much much better... parallax duplicates got almost zero... still a little blurry, still some shards but considerable improvement." `.ply`: `RUN9_30k.ply`.

## Run 10 — pose correction v2, more aggressive
Decaying LR: `lr_init=2e-3 → lr_final=2e-4`, exponential over full 30k.
**Result:** PSNR 15.41→15.79→15.97→**16.02**(peak@15000)→15.80→**15.59** — worse than v1 (16.81), peak-then-decline shape v1 didn't show.
**Verdict:** confirmed good by user despite lower PSNR than v1 — consistent "PSNR misleads" pattern. `.ply`: `RUN10_30k.ply`.

## Run 11 — pose correction v2, 60k iterations, 100% train (no held-out split)
**Date:** 2026-09-17 (new pod, A6000 49GB, VRAM unconstrained).
Changes: (1) `opt.iterations` 30000→60000; (2) fixed latent bug — `position_lr_max_steps` was hardcoded 30000, now tied to `opt.iterations`; (3) `dataset.eval` True→False (all 1653 frames).
**Result:** clean, no errors. No PSNR (no test set). `.ply`: `RUN11_60k.ply` (397,945 Gaussians).
**Verdict:** structurally clearer but more spike/shard artifacts than Runs 9/10. Cause: `densify_and_prune` pruning stops dead at fixed `densify_until_iter=15000` — 75% of a 60k run runs with zero pruning after that vs. 50% for a 30k run, a much larger unprotected window.
**Note:** 24GB VRAM cap added to `train.py` *after* this run — Run 11 ran unconstrained on the A6000's ~49GB.

## Run 12 — Run 11 config but 30k iterations, 24GB VRAM cap active
**Result:** completed cleanly under 24GB cap, no OOM — first run to validate the pipeline fits the real RTX 4090 deployment budget. `.ply`: `RUN12_30k.ply` (398,716 Gaussians).
**Verdict:** validates deployment-budget fit; visual verdict pending at the time (later: "same as 17" per consolidated review below).

## Labeling convention, starting Run 13 (2026-09-18)
`train.py` fix: `RUN_LABEL` names the wandb run and is written to `wandb.config`; `CHECKPOINT_DIR = f"checkpoints/{RUN_LABEL}"` auto-isolates checkpoints per run; `wandb.config` now built from every `opt`/`dataset`/`pipe` field automatically. Pod authenticated to `sarayusapa`'s wandb account — runs sync live from Run 13 on.

## Run 13 — pose correction constant LR, 60k iterations, needle-Gaussian pruning
Changes: (1) `iterations` 30000→60000; (2) pose-corr LR constant `lr_init=lr_final=5e-4` (not v2's decaying `2e-3`); (3) new needle-Gaussian pruning: sorts 3 scale axes descending, prunes where largest > 5x second-largest — verified in isolation: isotropic ratio 1.0 (kept), disk ratio 1.25 (kept), needle ratio 10.0 (pruned); (4) pruning (opacity+size+needle, via new `prune_only()`) now continues past `densify_until_iter=15000` for the entire run instead of stopping there; `max_radii2D` tracking also runs full duration.
**Result:** ⚠️ **Aborted by user at iteration 30000** (of 60000) — population collapsed:

| iteration | Gaussians |
|---|---|
| 3000 | 399,988 |
| 6000 | 399,741 |
| 12000 | 399,148 |
| **18000** | **91,198** |
| 24000 | 87,853 |
| 30000 | 84,209 |

77% of the population gone between 12000 and 18000 — a sudden cliff at the phase transition (continuous prune-only, no replenishment, every 100 iters), not gradual decay. Slower decline afterward (91,198→84,209, ~7.7% more).
**Root cause (diagnosed):** needle threshold 5.0 was validated only against synthetic extremes, never against a real converged scene's shape distribution — legitimate converged Gaussians commonly exceed a 5x ratio, so even a modest per-pass false-positive rate compounds catastrophically over ~150 zero-replenishment passes.
**Verdict:** user: **"extremely aggressive... cannot see a single gaussian in the renderer, they look like faint tiny points extremely sparse."** 84,209 Gaussians is nowhere near enough density. Explicit directive: cut threshold and cadence by an order of magnitude, not tune — target threshold 20-50+ (not 10-20), late-pass cadence every 1000-2000+ iters (not 100), consider a dry-run/logging-only pass before trusting the needle criterion live again. `.ply`: `RUN13_30k.ply` (84,209 Gaussians).

## Run 14 — scaled-down needle pruning
**Date:** 2026-09-18/19, new pod (RTX PRO 4500 Blackwell, torch upgraded to 2.11.0+cu128 for `sm_120` support; gsplat JIT ~122s one-time compile).
Changes: (1) `needle_ratio_threshold` 5.0→**30.0**; (2) new independent `opt.late_prune_interval=3000` (~15 passes vs. Run13's ~150).
**Result:** completed to 60000 (~15 min wall clock).

| iteration | Gaussians |
|---|---|
| 3000 | 400,019 |
| 6000 | 399,856 |
| 12000 | 399,540 |
| **18000** | **124,774** |
| 24000 | 100,460 |
| 30000 | 97,544 |
| 36000 | 95,340 |
| 42000 | 93,593 |
| 48000 | 92,304 |
| 60000 | 90,578 |

Same cliff location, similar final magnitude (90,578 vs. Run13's 84,209), but decelerating decline (-24314, -2916, -2204, -1747, -1289...) not one instant crash. `.ply`: `RUN14_18k.ply` (124,774), `RUN14_60k.ply` (90,578).
**Root cause (isolated on `checkpoint_12000.pth`):** `opacity<=0.002`: 0.0%. `needle_ratio>30.0`: 0.0% (median 1.16, p99 2.67, max 12.8 — threshold fix confirmed correct). `scaling.max>0.1*extent` (extent=2.843, threshold≈0.284; observed max 0.278): ~0%. None of the three scaled-down criteria explain the cliff — the real culprit is `max_radii2D>20px` (screen-space size), which accumulates max-ever-observed radius since the last prune call; widening `late_prune_interval` from 100→3000 stretched this accumulation window 30x, so far more Gaussians register a transient large radius from grazing-angle sampling and get falsely flagged.
**Verdict:** partial win — needle threshold/cadence fix confirmed non-issues (prune ~0%), but conflating that cadence with the screen-size check's reset window reintroduced comparable collapse via a different mechanism. See Run 15.

## Run 15 — decoupled prune cadences
`prune_only()` split: `prune_size_and_opacity()` every 100 iters (original cadence, continuous past `densify_until_iter`), `prune_needles()` separately every `late_prune_interval`(3000).
**Result:** hypothesis **not confirmed** — comparable collapse:

| iteration | Gaussians |
|---|---|
| 3000 | 399,999 |
| 6000 | 399,858 |
| 12000 | 399,534 |
| **18000** | **93,949** |
| 24000 | 90,850 |
| 30000 | 87,633 |
| 36000 | 84,342 |
| 42000 | 81,792 |
| 48000 | 79,793 |
| 54000 | 78,133 |
| 60000 | **76,691** |

Cliff magnitude 76.5% (vs. Run14's 68.8%, Run13's 77%); decline more sustained than Run14, ending lower (76,691 vs. 90,578). `.ply`: `RUN15_18k.ply` (93,949), `RUN15_60k.ply` (76,691).
**Revised diagnosis:** Runs 9-12 never ran the size/opacity check continuously with zero replenishment — only paired with simultaneous clone/split. Once decoupled, ~450 zero-replenishment passes (cadence 100) removed more than ~15 passes (cadence 3000, Run14) — **total pass count**, not per-pass threshold/cadence, is the dominant driver.
**Verdict:** ⚠️ still not usable. **User decision (2026-09-19):** cap total late-training prune passes to a small fixed count. See Run 16.

## Run 16 — capped total late-training prune passes (7 max)
New `opt.late_prune_max_passes=7`; both checks reunified on one cadence (`late_prune_interval=6000`); no further late pruning once cap hit.
**Result:**

| iteration | Gaussians | pass # |
|---|---|---|
| 3000 | 400,014 | (densify phase) |
| 6000 | 399,850 | (densify phase) |
| 12000 | 399,535 | (densify phase) |
| **18000** | **169,929** | 1 |
| 24000 | 112,305 | 2 |
| 30000 | 104,437 | 3 |
| 36000 | 102,243 | 4 |
| 42000 | 100,666 | 5 |
| 48000 | 99,425 | 6 |
| 54000 | 98,442 | 7 (cap) |
| 60000 | **98,442** | — (unchanged) |

`.ply`: `RUN16_60k.ply` (98,442).
**Diagnosis:** cap works — population provably bounded (54000→60000 identical). But the first pass alone (12000→18000) removes 57.5% in one shot, more than any single pass in Runs 14/15 — caused by the 15000-18000 gap being completely ungoverned (no clone/split, no pruning check) before the first late pass at 18000, letting `max_radii2D` accumulate unchecked for 3000 iters.
**Verdict:** ✅ real improvement — first run with a provably bounded population. Final: 98,442 (~25% of ~400k peak), not visually inspected yet.

## Late-training pruning: ABANDONED (2026-09-19)
User's verdict after seeing Runs 14/15/16: **"they all suck. we cant do pruning."** Three attempts (needle threshold → cadence → pass-count), each producing ~75-81% population loss (77k-98k Gaussians remaining from a ~400k peak) — structural, not a tuning problem: pruning after `densify_until_iter` has zero replenishment, so even a well-calibrated per-pass false-positive rate compounds given enough exposure. See `feedback_no_late_training_pruning.md`. **Going forward:** revert to Runs 9-12 behavior — pruning only during the active densify window, nothing after.

## Open question: has a 30k run with LR decay been tried without pruning?
Yes — **Run 12** (30k, pose-corr v2 decaying LR, 24GB cap, 100% train, standard pruning only during active window, predates Runs 13-16). `.ply`: `RUN12_30k.ply`. Never followed up visually — worth revisiting.

## Run 17 — Run 12's config + depth supervision explicitly confirmed active
Investigation found depth supervision was never actually off since Run 7 (`depth_l1_weight_init=1.0` nonzero in every commit; dataset's `depths/` folder has 1653 valid `.npz`, ~70% mask coverage, `depth_reliable=True` for essentially every frame) — it silently contributed to every run 7-16. This run makes it explicit (new `wandb.config["depth_supervision_dir_present"]` flag) and reproduces Run 12's config on the post-pruning-revert codebase.
**Result:** clean.

| iteration | Gaussians |
|---|---|
| 3000 | 399,935 |
| 6000 | 399,554 |
| 12000 | 399,168 |
| 18000 | 398,785 |
| 24000 | 398,785 |
| 30000 | 398,785 |

Stable/flat from 18000 (no clone/split or pruning after `densify_until_iter=15000`, exactly as designed). `.ply`: `RUN17_30k.ply` (398,785). **Verdict:** pending visual inspection.

## Run 18 — densify_until_iter removed (full-run densification), opacity_cull 0.002→0.008, needle threshold 30→25
Changes: (1) `densify_until_iter` tied to `iterations`(30000) — full-run densify+prune, paired with replenishment; (2) `opacity_cull` 0.002→0.008; (3) `needle_ratio_threshold` 30.0→25.0.
**Result:** no OOM (VRAM ~9.4GB/24GB), but population shrank continuously the entire run:

| iteration | Gaussians |
|---|---|
| 3000 | 296,919 |
| 6000 | 225,251 |
| 12000 | 157,579 |
| 18000 | 123,709 |
| 24000 | 102,740 |
| 30000 | **88,921** |

Diverges from every prior run (9-17, all ~399-400k@12000) almost immediately; final 88,921 lower than any abandoned late-pruning run (14: 90,578; 15: 76,691; 16: 98,442) despite active replenishment.
**Diagnosis:** likely dominant cause `opacity_cull` 4x increase; full-run densification window (doubling pruning-cycle exposure) not isolated.
**Verdict:** ⚠️ opposite of intended effect (wanted more Gaussians, got the lowest count of the session). `.ply`: `RUN18_30k.ply` (88,921).
**User feedback:** spikes improved, detailing lacking. Asked for `opacity_cull→0.004`, 60k run. Also fixed a latent bug: `opacity_reset_interval` was fixed `40000` (would silently re-enable at 60k iters and re-trigger the white-crystal failure mode) → tied to `self.iterations+1`.

## Run 19 — opacity_cull dialed back to 0.004, 60k iterations
**Result:** still shrinks continuously, ends lower than Run 18:

| iteration | Gaussians |
|---|---|
| 3000 | 376,279 |
| 6000 | 310,862 |
| 12000 | 197,829 |
| 18000 | 146,964 |
| 24000 | 120,990 |
| 30000 | 105,501 |
| 36000 | 95,291 |
| 42000 | 88,402 |
| 48000 | 83,480 |
| 54000 | 79,910 |
| 60000 | **77,431** |

Better start than Run18 (376,279 vs. 296,919@3000 — halving `opacity_cull` slowed early erosion) but ends lower (77,431 vs. 88,921) since twice as long (~600 vs. ~300 densify/prune cycles).
**Diagnosis:** `opacity_cull` alone doesn't stop the shrink, only its rate. Dominant variable: `densify_until_iter` spanning the full run — removal outpaces growth (`densify_grad_threshold=0.00008`) regardless of length.
**Verdict:** ⚠️ still shrinking, below Run17's stable 398,785 baseline and below Run18. Untried levers: (1) bounded window, (2) lower `densify_grad_threshold`.
**User's direct visual read:** "i see many many hazy gaussians and a lot of details here and there missing... views that werent covered much... theyve all gone." Continuous full-run pruning disproportionately erodes under-observed regions. Decision: revert to bounded window; pursue detail via `densify_grad_threshold`. See Run 20.

## Runs 20-23 — bounded-window + densify_grad_threshold/opacity_cull crash arc (all CRASHED, root cause ultimately infrastructure, not hyperparameters)
Four runs, each changing one lever and crashing identically (`checkpoint_3000.pth` corrupted or missing, zero traceback, GPU memory cleanly freed): Run20 (`densify_until_iter=50%`, `densify_grad_threshold` 0.00008→0.00004, `opacity_cull`→0.003), Run21 (`densify_grad_threshold`→0.00006, same window/cull), Run22 (window→65%, `densify_grad_threshold` reverted to stable 0.00008, `opacity_cull` still 0.003). A 5-run comparison table briefly implicated `opacity_cull=0.003` as the common factor in every crash (vs. Run18/19's safe 0.008/0.004):

| run | window | grad_threshold | opacity_cull | result |
|---|---|---|---|---|
| 18 | 100% | 0.00008 | 0.008 | safe (88,921) |
| 19 | 100% | 0.00008 | 0.004 | safe (77,431) |
| 20 | 50% | 0.00004 | 0.003 | crashed |
| 21 | 50% | 0.00006 | 0.003 | crashed |
| 22 | 65% | 0.00008 | 0.003 | crashed |

Run23 (`opacity_cull` reverted to the proven-safe 0.004, window still 65%) **crashed again anyway**, retracting that theory. Instrumented with `mem_monitor.sh` (2s polling): GPU memory rose smoothly ~9.3GB→10GB (of 24GB cap), container RAM ~7.8GB→8.75GB (of ~87.5GB cgroup limit, `93,999,996,928` bytes) — both normal, no spike; process simply vanished between samples. `/sys/fs/cgroup/memory.events` showed `oom 0, oom_kill 0, oom_group_kill 0` all session; `nvidia-smi -q` showed zero ECC errors/retired pages. A retry on the same pod crashed identically (5th crash in a row).
**Final understanding: root cause was pod infrastructure (preemption/reclaim or a shared-host OOM bypassing cgroup accounting), not `densify_grad_threshold`, `opacity_cull`, or window size** — confirmed later by Runs 24-31 running the same/more-aggressive values cleanly on a new pod. Moved to a fresh pod.

## Runs 24-26 — controlled parallel comparison: densify window duration (50%/65%/80%)
**Date:** 2026-09-19, new pod: single A100-SXM4-80GB (80GB VRAM, 2TB host RAM, no cgroup memory limit — vs. the prior pod's ~87.5GB hard cap, likely explaining its unreliability). Hit the network volume's own storage quota mid-setup; freed space by deleting Runs 13-16/18/23 checkpoint dirs (kept `.ply` exports).
Setup: 3 identical copies except `densify_until_iter = int(iterations*0.50/0.65/0.80)`. Fixed at Run19-safe values: `densify_grad_threshold=0.00008`, `opacity_cull=0.004`.
**Launch hiccup:** gsplat CUDA JIT-compile race across the 3 simultaneous launches — `run25`/`run26` won (~190s compile), `run24` lost (missing `.so`), fixed by restarting `run24` after the cache was warm.
**Result:** all 3 clean, no crashes.

| iteration | Run24 (50%) | Run25 (65%) | Run26 (80%) |
|---|---|---|---|
| 3000 | 377,542 | 380,943 | 377,474 |
| 6000 | 314,461 | 319,747 | 316,698 |
| 12000 | 203,291 | 205,408 | 199,221 |
| 18000 | 148,315 | 149,753 | 146,876 |
| 24000 | 120,902 | 121,805 | 121,041 |
| 30000 | 105,166 | 106,358 | 105,984 |
| 36000 | **105,166** (frozen) | 96,156 | 95,857 |
| 42000 | 105,166 | **92,349** (frozen) | 88,829 |
| 48000 | 105,166 | 92,349 | **83,668** (frozen) |
| 54000-60000 | 105,166 | 92,349 | 83,668 |

Each population freezes exactly at its own `densify_until_iter` (30000/39000/48000), confirming the "pruning stops dead, nothing changes after" design works.
**Critical finding: longer window → LOWER final population, monotonically** (105,166 / 92,349 / 83,668) — all three follow the *same* shrinking trajectory while active. **This falsifies the premise behind Runs 18-19:** the net dynamic during the active window (grad_threshold=0.00008 paired with opacity_cull=0.004) is net-shrinking throughout, not net-growing; a longer window just lets that same shrink run further before freezing at a lower floor. Runs 18/19's 100%-window results (88,921/77,431) are simply further points on this same curve.
**Verdict:** ⚠️ window duration was the wrong lever entirely. The actual lever is the growth/removal *balance* (lower `densify_grad_threshold` relative to `opacity_cull`) — what Runs 20-22 attempted; those crashes are now known (Run23's instrumentation + this pod's headroom) to be the old pod's infrastructure problem, not real training instability. `.ply`s: `RUN24_iter60000_densify50pct.ply`(105,166), `RUN25..._65pct.ply`(92,349), `RUN26..._80pct.ply`(83,668).

**User's visual verdict: "too many spikes in all 3."** Investigated Run25's `checkpoint_60000` (92,349 Gaussians): needle-ratio distribution — median 1.55, p90 3.78, p95 5.12, p99 9.41, p99.9 20.34, max 59.46. At the active threshold 25.0, only **40 Gaussians (0.04%)** exceeded it — the needle criterion has been essentially inert since Run 13, explaining why spikes were never cleaned up. Also: whichever window is chosen, the population freezes completely at that cutoff — any spikes present at that moment are permanent.

## Run 27 — functional needle threshold (10), higher opacity_cull (0.005), full 100% window
Changes: (1) `needle_ratio_threshold` 25.0→**10.0** (catches genuine tail outliers, ~0.85% at the iter60000 checkpoint measured, without touching the bulk median 1.55/p90 3.78); (2) `opacity_cull` 0.004→**0.005**; (3) `densify_until_iter` 65%→**100%** — paired with clone/split replenishment the whole run (not the abandoned zero-replenishment mechanism), so the now-functional needle check gets continuous opportunity to clean spikes as they form.
**Result:** clean, no crashes.

| iteration | Gaussians |
|---|---|
| 3000 | 349,837 |
| 6000 | 273,565 |
| 12000 | 172,868 |
| 18000 | 130,550 |
| 24000 | 107,046 |
| 30000 | 92,635 |
| 36000 | 82,797 |
| 42000 | 76,034 |
| 48000 | 70,963 |
| 54000 | 67,191 |
| 60000 | **64,395** |

Declines continuously (never freezes) — lowest final count of the session, expected given all three changes compound toward removal. `.ply`s: `RUN27_iter30000_needle10_opacitycull005.ply`(92,635), `RUN27_iter60000...ply`(64,395).
**Verdict:** pending visual inspection — goal was fewer spikes, accepting a lower Gaussian count as tradeoff.

## User's consolidated visual review, Runs 9-27 (2026-09-19)
Direct free-navigation comparison of every renamed `.ply` (`RUNxx_Nk.ply`):

| run | verdict |
|---|---|
| 9 | very blurry but structurally much better |
| 10 | less blurry than 9 |
| 11 | much less blurry but a lot of spikes |
| 12 | slightly less spiky than 11 |
| 13 | nothing visible |
| 14 | almost nothing visible |
| 15 | nothing visible |
| 16 | almost nothing visible |
| 17 | same as 12 |
| 18 | almost same as 17 |
| 19 | is 18 but a lot of tiny spikes |
| 24 | too many spikes |
| 25 | again spikes but slightly clearer features |
| 26 | decent clarity but spiky |
| 27 | quite blurry |

Runs 1-8 (pre-pose-correction): **"horrible."** Run 9 is the real turning point of the session.

**Synthesis — an unresolved blur/spike tradeoff:**
- Runs 13-16 (needle-collapse, ~64k-125k Gaussians) → "nothing/almost nothing visible" — matches counts exactly.
- Runs 11/12/17/18 → clear/sharp but spiky. Run 19 (`opacity_cull` 0.008→0.004) made spikes worse ("a lot of tiny spikes") — first direct confirmation lowering `opacity_cull` increases visible spike density.
- Runs 24-26 (50/65/80% window) all spiky; **26 (80% window) stands out: "decent clarity but spiky"** — best clarity/spike balance so far.
- **Run 27 (functional needle=10, opacity_cull=0.005, 100% window) overcorrected into "quite blurry"** — opposite failure mode, structurally similar to 13-16's pattern, less catastrophic in degree.

**Open problem:** every spike-reduction attempt via pruning (13-16, 27) traded spikes for blur/sparsity; every config that stays clear (11, 12, 17-19, 24-26) keeps spikes. Worth isolating Run27's three simultaneous changes (e.g. take Run26's config, add only the needle-threshold fix).

## Run 28 — differentiable anisotropy loss + LR decay for all Gaussian params
**User direction:** rejected COLMAP BA; try preventing needle-shaped Gaussians during training (a loss term) rather than pruning after the fact, given every prune-based attempt (13-16, 27) traded spikes for blur once pushed hard. Separately: noisy losses → decay LR for all Gaussian params (previously only `xyz` decayed).
Changes from Run26: (1) new `GaussianModel.anisotropy_loss()` — `relu(largest/2nd-largest_scale_ratio - threshold).mean()`, `anisotropy_loss_weight=0.02`, `anisotropy_ratio_threshold=5.0`; (2) `needle_ratio_threshold` reverted to inert 25.0 (isolate the soft loss alone); (3) `opacity_cull` reverted to 0.004; (4) `densify_until_iter` reverted to Run26's 80% window; (5) new LR decay for `f_dc`/`f_rest`/`opacity`/`scaling`/`rotation` — each decays to **1/10th** of init over the full run.
**Crash saga (root cause NOT a training bug):** run died silently at iter 36000 twice, zero traceback (checkpoint healthy: no NaN/Inf, `anisotropy_loss=0.00035`). Ruled out stale JIT cache, shell/session issues, gsplat itself (362s fresh compile succeeded). **Actual root cause** (found running in foreground, unbuffered): `OSError: [Errno 122] Disk quota exceeded` — network volume filled to 43GB again, which silently broke the `nohup` log redirect (producing the "empty log" signature) and wandb's own logging. Freed space (deleted superseded checkpoints for Runs 17/19/24-27, kept `.ply` exports), verified with a write test, resume completed cleanly.
**Resume support added** (`train.py`): `checkpoint_path` now actually used (was hardcoded `None`); `torch.load(weights_only=False)`. **Known gap:** `pose_corr`'s per-camera deltas are not serialized/restored on resume — reset to zero-init on resume (acceptable here, needs fixing for regular use).
**Result:** resumed cleanly from 36000 to 60000.

| iteration | Gaussians |
|---|---|
| 36000 | 133,117 |
| 42000 | 128,873 |
| 48000 | **126,015** (frozen, = densify_until_iter=48000) |
| 54000 | 126,015 |
| 60000 | 126,015 |

Final population (126,015) notably HIGHER than Run26's 83,668 despite identical window/opacity_cull — only diffs are anisotropy loss + full LR decay, not isolated which. `.ply`: `RUN28_60k.ply` (126,015).
**User's verdict:** "its kinda better yes but maybe i think we should slightly increase opacity cul." Confirms a real, modest improvement. Also asked to double `zfar` (`far_margin` 1.5→3.0) — far-plane culling suppresses far-away detail at both render and densification-gradient level. See Run 29.

## Run 29 — opacity_cull nudged to 0.0045, zfar doubled
Changes: (1) `opacity_cull` 0.004→**0.0045**; (2) `far_margin` 1.5→**3.0** — confirmed via wandb: `zfar` ~5.75→**11.496** (`znear` unchanged ~0.00488).
**Result:** clean, no crashes.

| iteration | Gaussians |
|---|---|
| 3000 | 369,539 |
| 6000 | 303,645 |
| 12000 | 208,659 |
| 18000 | 165,411 |
| 24000 | 143,066 |
| 30000 | 130,596 |
| 36000 | 123,282 |
| 42000 | 118,727 |
| 48000 | **115,721** (frozen, = densify_until_iter=48000) |
| 54000 | 115,721 |
| 60000 | 115,721 |

Final 115,721 slightly lower than Run28's 126,015, consistent with the modest `opacity_cull` bump. `.ply`: `RUN29_60k.ply` (115,721).
**User's takeaway after Runs 28-29:** spikes largely resolved, blur is the remaining bottleneck. Suggested higher LR. Diagnosis: Run28's LR decay quiets refinement capacity right in the last 20% of the run, when densification has already stopped. Candidates ranked: (1) gentler decay, (2) lower `depth_l1_weight_init` (currently 1.0), (3) revisit `densify_grad_threshold` (now known the crashes were pod-infra), (4) toggle antialiasing off. User chose (3) first.

## Run 30 — densify_grad_threshold lowered to 0.00006, finally a clean test
Only change: `densify_grad_threshold` 0.00008→0.00006 (the value Run21 crashed with — now known unrelated).
**Result:** clean.

| iteration | Gaussians |
|---|---|
| 3000 | 365,954 |
| 6000 | 301,434 |
| 12000 | 207,746 |
| 18000 | 164,766 |
| 24000 | 143,095 |
| 30000 | 130,837 |
| 36000 | 123,667 |
| 42000 | 119,199 |
| 48000 | **116,285** (frozen) |
| 54000 | 116,285 |
| 60000 | 116,285 |

Final 116,285 nearly identical to Run29's 115,721 (<0.5% diff) despite a 25% more aggressive trigger. `.ply`: `RUN30_60k.ply` (116,285).
**Diagnosis:** lowering `densify_grad_threshold` alone doesn't change final count — more Gaussians spawn early but get pruned just as fast.
**Verdict:** confirmed safe, but ineffective alone for "more detail." **User:** "not that much of a visible difference... needs a more considerable change" — asked for 0.00004. See Run 31.

## Run 31 — densify_grad_threshold pushed to 0.00004 (50% reduction)
**Result:** clean — second consecutive clean run at a threshold that crashed twice on the old pod.

| iteration | Gaussians |
|---|---|
| 3000 | 366,685 |
| 6000 | 298,702 |
| 12000 | 203,972 |
| 18000 | 160,230 |
| 24000 | 138,918 |
| 30000 | 126,888 |
| 36000 | 119,960 |
| 42000 | 115,564 |
| 48000 | **112,776** (frozen) |
| 54000 | 112,776 |
| 60000 | 112,776 |

Final 112,776 is LOWER than both Run29 (115,721) and Run30 (116,285) despite the most aggressive trigger — not monotonic; all three (115,721/116,285/112,776 for 0.00008/0.00006/0.00004) within noise. `.ply`s: `RUN31_30k.ply`(126,888), `RUN31_60k.ply`(112,776).
**Verdict:** ⚠️ `densify_grad_threshold` definitively NOT the lever for more Gaussians across a 2x range. Real bottleneck: `opacity_cull` and/or `anisotropy_loss_weight`.
**User:** "bro then what tf do we do now." Key realization: population flat ~112k-126k across Runs 28-31 regardless of threshold, yet the complaint shifted from spikes to BLUR starting exactly at Run28 (which introduced anisotropy loss + LR decay; Run26, without either, wasn't blurry). Direction: reverse Run28's two additions partway. See Run 32.

## Run 32 — anisotropy loss cut 4x, LR decay loosened, threshold reverted to neutral
Changes: (1) `anisotropy_loss_weight` 0.02→**0.005**; (2) `*_lr_final` loosened from 1/10th of init to **1/3rd**; (3) `densify_grad_threshold` reverted 0.00004→**0.00008** (ruled out as a lever). `opacity_cull`(0.0045) and doubled `zfar`(11.5) kept from Run29.
**Result:** clean.

| iteration | Gaussians |
|---|---|
| 3000 | 363,557 |
| 6000 | 299,353 |
| 12000 | 195,769 |
| 18000 | 147,309 |
| 24000 | 123,660 |
| 30000 | 109,696 |
| 36000 | 100,783 |
| 42000 | 94,973 |
| 48000 | **90,959** (frozen) |
| 54000 | 90,959 |
| 60000 | 90,959 |

Final 90,959, notably lower than Run29's 115,721 despite identical `opacity_cull`/threshold — only diffs are the lower anisotropy weight and looser decay. `.ply`: `RUN32_60k.ply` (90,959).
**User's verdict:** Run32's 60k clearly clearer than Run32's 30k (expected), but barely different from Run31's 60k — rules out the anisotropy-cut/LR-loosening as the main blur fix. Moved to depth supervision. See Run 33.

## Run 33 — depth supervision disabled entirely (isolated test)
Only change: `depth_l1_weight_init/final` 1.0/0.01→**0.0/0.0** (loss returns exactly 0, never computed).
**Result:** clean.

| iteration | Gaussians |
|---|---|
| 3000 | 387,793 |
| 6000 | 362,351 |
| 12000 | 304,298 |
| 18000 | 258,715 |
| 24000 | 226,630 |
| 30000 | 206,040 |
| 36000 | 192,867 |
| 42000 | 184,112 |
| 48000 | **178,174** (frozen) |
| 54000 | 178,174 |
| 60000 | 178,174 |

Final 178,174 — dramatically higher than every run since Run28 (90,959-126,015), 40-95% more Gaussians survived. `.ply`: `RUN33_60k.ply` (178,174).
**User:** "bro figuratively nothing has been changing on the surface" — despite population going from ~90k to ~178k across Runs 28-33, none of it visibly changed sharpness. **This was the actual turning point:** checked the input pipeline directly and found every training image this entire session was **672x378**, not native 1920x1080 — `loger_to_colmap.py` had been saving LoGeR's own downsampled inference input (`PIXEL_LIMIT=255000`) as the photometric target instead of the real video frame. No loss-weight tuning can sharpen detail discarded before training ever saw it.

## The resolution fix: native 1920x1080 training images
**Root cause verified:** source video native 1920x1080; images had been 672x378 the entire session (2.86x downsample). Confirmed exact frame correspondence for i=0-1652 (LoGeR drops the trailing 15 frames) via pixel-diff at 6 sample indices using sequential reads (`cv2` `CAP_PROP_POS_FRAMES` seeking confirmed unreliable — failed to seek to frame 1652, wrong results at 500/1000).
**Fix (`loger_to_colmap.py`):** new `--source_video` arg reads the video sequentially in lockstep, saving the native frame as the training image; intrinsics (fx/fy/cx/cy) scaled by resolution ratio; hard abort if aspect ratio mismatches LoGeR's inference resolution by >1%. Point cloud RGB/depth export left at low-res.
**Dataset regenerated:** `data/inference_run_native/` (6.3GB vs. 622MB). `fx` 286.75→**819.28**, `cx`/`cy`→**960/540**. All 1653 frames processed, zero dropped.
**Second bug found (`colmap_loader.py`):** `load_colmap_scene` eagerly moved every image to GPU — fine at low-res (~5GB for 1653 images) but native res needs **~41GB**, blowing the 24GB budget (Run34's first attempt OOM'd: "Tried to allocate 24.00 MiB" against a full 24GB cap). **Fixed:** images stay CPU-resident at load, moved to GPU lazily per-sample in `train.py`. Side benefit: VRAM usage dropped dramatically (Run34's retry used only ~1.6-2GB well into training on an 80GB A100).
**"COLMAP" naming clarified:** `loger_to_colmap.py` runs no COLMAP algorithm — only `colmap_refine.py` does (unused since Run8's revert). "COLMAP" here means only the binary file format `train.py`'s loader reads.

## Run 34 — first validation run at native resolution (sweep arm A)
Spans 3 wandb runs across 2 pods (first attempt OOM'd during scene load; A100 pod reached checkpoint_30000 before being reclaimed overnight; resumed on a new L40 46GB pod).
Config: identical to Run33's recipe (no depth sup, `anisotropy_loss_weight=0.005`, LR decay to 1/3, `opacity_cull=0.0045`, `densify_grad_threshold=0.00008`, 80% window, doubled `zfar`) — only the dataset changed to `inference_run_native`.
**Checkpoint policy changed:** saves only at midpoint and end (30000/60000 for a 60k run) — cuts disk usage ~5.5x.
**Result:** clean after the OOM fix + pod-switch resume.

| iteration | Gaussians |
|---|---|
| 30000 | 206,533 |
| 60000 | **178,329** |

`.ply`s: `RUN34_30k.ply`(206,533), `RUN34_60k.ply`(178,329).
**User's visual read of the 30k checkpoint: still very blurry**, despite GT images confirmed at full native quality (1920x1080 via wandb `render/gt` dims). Sobering — removing the resolution ceiling didn't fix blur alone. Two live hypotheses: (1) 60k needs more refinement time, or (2) blur comes from elsewhere (LoGeR pose/pointmap accuracy, anisotropy loss, insufficient densification) and resolution was necessary but not sufficient.

## N1-N8: point-cloud-prior-strength factorial sweep
**Date:** 2026-09-27. **Trigger:** after the loss-noise-floor diagnosis, make the trusted LoGeR point-cloud prior more directly influence the optimized Gaussians. Base recipe (all 8 arms): `densify_until_iter=100%`, `anisotropy_loss_weight=0`, 60k iters, non-BA native-res dataset, pose correction unmodified.

Three mechanisms, full 2^3 factorial:
- **D (depth supervision):** re-enabled (`1.0/0.01`).
- **A (anchor loss, new):** L2 tether of each Gaussian's current position to `xyz_init`, propagated through the densify/prune lifecycle. `anchor_loss_weight` 0.0→1.0 when on — not empirically calibrated, a starting point.
- **M (median-distance extent, new):** `use_point_density_extent` False→True — replaces `cameras_extent` with `median_nn_distance * point_density_extent_scale` (default 70.0, calibrated from `cameras_extent≈2.84`/`median_nn_distance≈0.04`) as the densify/prune reference scale. Does not touch `spatial_lr_scale`.

| Arm | D | A | M | RUN_LABEL |
|---|---|---|---|---|
| N1 | off | off | off | sweepN1_base |
| N2 | on | off | off | sweepN2_depth |
| N3 | off | on | off | sweepN3_anchor |
| N4 | off | off | on | sweepN4_medianextent |
| N5 | on | on | off | sweepN5_depth_anchor |
| N6 | on | off | on | sweepN6_depth_medianextent |
| N7 | off | on | on | sweepN7_anchor_medianextent |
| N8 | on | on | on | sweepN8_all |

**User's visual verdict on I and J's 30k checkpoints** (I=plain F+B recipe, J=Run12+anisotropy, neither has D/A/M active): "both of them are really really fuzzy and extremely blurry. structurally okay which is why loss is at like 0.1, but very blurry." `.ply`s: `SWEEPI_30k.ply`(206,346), `SWEEPJ_30k.ply`(397,990).

**I, J, N1 final (60k):**

| Arm | loss | l1 | ssim | Gaussians | max_scale | anchor_loss |
|---|---|---|---|---|---|---|
| I (F+B) | 0.1375 | 0.0756 | 0.615 | 172,948 | 0.293 | n/a |
| J (Run12+aniso) | 0.1157 | 0.0616 | 0.671 | 397,990 | 1.376 | n/a |
| N1 (control) | 0.1251 | 0.0829 | 0.706 | 173,704 | 0.297 | 0.00212 |

N1 validates the new code has no regression (all-off matches I's ~173k Gaussians, max_scale ~0.29-0.30); `anchor_loss=0.00212` at weight=0 confirms the metric computes correctly (natural drift), mechanism live but inert. `.ply`s: `SWEEPI_60k.ply`, `SWEEPJ_60k.ply`, `SWEEPN1_30k.ply`, `SWEEPN1_60k.ply`.

**N2/N3/N4 at 30k:**

| Arm | loss | l1 | ssim | Gaussians | max_scale | anchor_loss |
|---|---|---|---|---|---|---|
| N2 (depth) | 0.144 | 0.089 | 0.763 | 123,824 | 0.277 | 0.0053 |
| N3 (anchor) | 0.145 | 0.103 | 0.691 | 171,248 | 0.288 | 0.00034 |
| N4 (median-extent) | 0.232 | 0.176 | 0.544 | 145,144 | **0.063** | 0.0017 |

N4's `max_scale` dramatically smaller (0.063 vs. ~0.28) — controlling floater size aggressively at cost to fit quality. `.ply`s: `SWEEPN2_30k.ply`, `SWEEPN3_30k.ply`, `SWEEPN4_30k.ply`.

**N2/N3/N4 final (60k):**

| Arm | loss | l1 | ssim | Gaussians | max_scale | anchor_loss |
|---|---|---|---|---|---|---|
| N2 (depth) | 0.129 | 0.078 | 0.674 | 102,508 | 0.285 | 0.0056 |
| N3 (anchor) | 0.141 | 0.082 | 0.622 | 146,238 | 0.293 | 0.00036 |
| N4 (median-extent) | 0.153 | 0.111 | 0.677 | 99,179 | **0.047** | 0.0016 |

N4's max_scale stayed small (0.047); l1/ssim caught up to tie N2 on ssim by 60k despite being worst at 30k. `.ply`s: `SWEEPN2_60k.ply`, `SWEEPN3_60k.ply`, `SWEEPN4_60k.ply`.

**User's visual verdict on N1-N4's 60k (2026-09-28):** "N2 N3 60k decent. N3 a little more noisy and more spiky, N1 better but still scope for improvement. N4 has no details?? like its so sparse i can hardly see anything. so lets scrap N4... N2 so far best outputs N3 maybe more modifs."

**N4 scrapped.** "Sparse, no detail" matches the max_scale data exactly (0.047-0.063 vs. ~0.28-0.29 for N1/N2/N3) — `point_density_extent_scale=70.0` was too aggressive, killing legitimate large/flat-surface Gaussians along with real floaters. **Drop M from future N5-N8 launches** — N6/N7 (involving M) not worth running at this calibration; would need a real calibration pass (binary-search the scale against max_scale staying ~0.2-0.3) if revisited.

**Ranking: N2 (depth) best so far; N3 (anchor) promising, needs tuning (`anchor_loss_weight=1.0` also an untuned guess); N1 (control) decent baseline.** Next priority: N5 (depth+anchor).

## N11/N12/N13: anchor_loss_weight sweep + extended-duration test
**Date:** 2026-09-29, new pod L40 46GB. **Trigger:** N5 (depth+anchor@1.0) was strongest from N1-N10; user wanted to test whether the noisy-but-decreasing loss converges with more iterations ("i want it to overfit"), and whether `anchor_loss_weight` should exceed 1.0 (0.3 was worse in N9/N10).

N11: N5's recipe, extended to 120k, checkpoints 30/60/90/120k. N12: N5's recipe, `anchor_loss_weight=1.5`, standard 60k. N13: N5's recipe, `anchor_loss_weight=2.0`, standard 60k.

| Arm | Config | loss | L1 | SSIM | Gaussians | max_scale |
|---|---|---|---|---|---|---|
| N5 (ref, 60k) | anchor@1.0 | **0.088** | **0.065** | **0.834** | 47,705 | 0.293 |
| N11 (120k) | anchor@1.0, 2x dur | 0.160 | 0.083 | 0.552 | 23,686 | 0.283 |
| N12 (60k) | anchor@1.5 | 0.144 | 0.088 | 0.647 | 42,109 | 0.300 |
| N13 (60k) | anchor@2.0 | **0.076** | **0.047** | **0.819** | 40,952 | 0.286 |

**Extending duration made it WORSE — falsifies the "needs more iterations to converge" hypothesis.** N11 (same recipe as N5, only duration changed) ended behind on every metric: loss nearly doubled (0.088→0.160), ssim dropped sharply (0.834→0.552), population eroded steadily (47,705@N5's-60k → 29,238@N11's-own-60k → 25,149@90k → 23,686@120k, roughly halved). Third independent confirmation duration isn't a lever for this recipe.
**anchor_loss_weight sweep real but NOT monotonic:** 1.0 (N5) and 2.0 (N13) both strong, nearly tied (ssim 0.834 vs. 0.819); 1.5 (N12) a clear dip (0.647) between them.
`.ply`s: `SWEEPN11_{30,60,90,120}k.ply`, `SWEEPN12_{30,60}k.ply`, `SWEEPN13_{30,60}k.ply`.
**User's visual verdict on N11's 90k: "really blurry so n11 sucks"** — matches ssim 0.552 vs. N5's 0.834.
**User's visual verdict on N5 vs. N13 (2026-09-30): "N5 is actually at par with N13."** Matches the numbers (N13 wins loss/L1: 0.076/0.047 vs. N5's 0.088/0.065; N5 edges ssim 0.834 vs. 0.819). Sharpens the curve shape: a plateau of comparably-good results at both ends (1.0, 2.0) with a real trough at 1.5 (N12, 0.647) — N5 and N13 are functionally interchangeable as the best 60k reference.

**Session resumed 2026-09-28** (new pod, A100 80GB, `ssh.runpod.io` proxy — requires `-tt` PTY, no SFTP/SCP, used base64-over-stdin transfer instead).
N5, N9 (anchor alone, weight=0.3), N10 (depth+anchor, weight=0.3) launched to test recalibrating `anchor_loss_weight` down. M dropped per N4 verdict. **Batch hit friction and was killed before any checkpoint completed (2026-09-28, user request):** the gsplat JIT-compile race recurred (N5/N10 crashed, N9 won and was used to relaunch the others); N9 also hung 1h13m at 0% GPU util and was restarted; N10 crashed a 2nd time on an unrelated `getcwd()` error (not root-caused). **Real finding:** 3-way concurrent dataset loading was abnormally slow (30-40+ min vs. usual ~20-25) — `vmstat` showed 6-10 processes blocked in I/O wait despite fast underlying storage (`dd`: 103 MB/s); bottleneck is **per-file-open latency on the network-mounted filesystem** (`mfs#us-ks-2.runpod.net`) compounding across 3x concurrent opens of ~1650 files. Dropping to 2-way concurrency fixed it. **Lesson: stagger dataset loading, cap concurrent loaders at 2** (not just 3 total trainers). Kill-time state: N5 training (no checkpoint), N9 compiling, N10 dead — nothing lost, configs persist on volume.
**Session paused at user request before completing the full factorial.** N5/N6/N7/N8 built (`/workspace/sweep/{N5,N6,N7,N8}`) but never launched — N8 (all three combined) is the actual point of this investigation line. Also added, not yet deployed: `loss_contrib/*` wandb logging (per-term weighted contribution: l1, dssim, depth, anisotropy, anchor) — deploy to N5-N8 when launched.

**Operational note:** disk-quota-exceeded again — `pod_results/` grew to ~1.5GB (never auto-cleaned), pushing `/workspace` to ~51GB against the ~50GB quota; `checkpoint_to_ply.py` failed for N3/N4's 60k exports (`OSError: Disk quota exceeded`, silently writing 0-byte files). Fixed by deleting already-pulled `.ply`s (freed ~2GB). Second time this exact quota ceiling hit this session — check `du -sh /workspace` before assuming an export failure is a code bug.

## N14/N15/N16: LR experiments (decay strength, init value, no-decay) — built on N13 (anchor@2.0)
**Date:** 2026-09-29. **Trigger:** user's theoretical read that the noisy-but-downward loss on N12/N13 is structural (each step targets a different GT frame), not estimation noise.

N14: N13's recipe, LR decay strengthened 3x→10x, extended to 120k (checkpoints 30/60/90/120k). N15: N13's recipe, init LRs doubled (`scaling_lr` 0.005→0.01, `rotation_lr` 0.001→0.002, `opacity_lr` 0.05→0.1, `feature_lr` 0.0025→0.005), same 3x decay ratio, 120k. N16: N13's recipe, decay removed entirely (flat LR), standard 60k.

**N14 Gaussian counts (30/60/90/120k):** 43,937 / 27,548 / 24,113 / 22,860 — steady erosion, no cliff. (Wandb's `train/loss` summary query returned empty — a metrics-API gap, not a training failure.)
**N15 Gaussian counts (30/60/90k, killed before 120k):** 34,963 / 20,084 / 13,753 — eroding faster than N14. Real wandb-confirmed transient loss spikes: `l1_loss` jumped **0.060→0.341** in one step (step 97930→98010), earlier spike to **0.354** at step 85340. **Killed by user instruction after visual inspection ("ok no this is really bad nuke it")** before 120k.
**N16 vs. N13 (decay is the only variable):**

| | 30k Gaussians | 60k Gaussians |
|---|---|---|
| N13 (3x decay) | 55,585 | 40,952 |
| N16 (no decay) | 47,001 (-15%) | 32,948 (-20%) |

**User's visual verdict on N16:** "decent structure and consistency, but a lot of features have gotten pruned, some still skewed/blurry/spiky." Confirms quantitatively — no decay keeps opacity/scaling/rotation LR at full strength, so pruning keeps firing aggressively through 60k instead of settling; same full-strength late LR is the likely source of the skewed/spiky artifacts too (same failure family as N14/N15, milder).
**Verdict: decay is load-bearing, not cosmetic** — it protects already-formed structure from late-training pruning/destabilization. **N13 (weight=2.0, standard LR, 3x decay) remains the strongest recipe** — higher LR (N14, N15) and no decay (N16) are all worse. `.ply`s: `SWEEPN14_{30,60,90,120}k.ply`, `SWEEPN15_{30,60,90}k.ply`(stopped 90k), `SWEEPN16_{30,60}k.ply`.

## N17/N18/N19/N20: anchor_loss_weight sweep at 120k (1.25/1.5/1.75/2.0)
**Date:** 2026-09-29/30. **Trigger:** characterize the shape between N5(1.0)/N12(1.5)/N13(2.0) at 120k — does higher anchor pull counteract the erosion that sank N11?
Config: all cloned from N13; N17=1.25, N18=1.5, N19=1.75, N20=2.0 (N13 rerun@120k). N11 (weight=1.0@120k, ssim 0.552) stands in as the low end, not rerun.
**Concurrency finding:** this pod (L40 46GB, 256 cores) couldn't sustain 4 concurrent 120k trainers — lost a process twice (once N18+N20 together, once N17 alone), zero trace (OOM suspected, not provable). 3 concurrent held rock-solid. **Established ceiling: 3 concurrent trainers for this pod class.**
**Interruption:** pod killed mid-run (user request). N19 was the casualty — died once during the 4-way instability, never relaunched, zero checkpoints. N17/N18/N20 completed cleanly to 120k unattended and were recovered from the persistent volume on a new A100 80GB pod.

**Results (120k final):**

| Arm | Weight | Duration | loss | L1 | SSIM | Gaussians |
|---|---|---|---|---|---|---|
| N11 | 1.0 | 120k | 0.160 | 0.083 | 0.552 | 23,686 |
| N17 | 1.25 | 120k | 0.125 | 0.068 | 0.668 | 22,346 |
| N18 | 1.5 | 120k | 0.130 | 0.086 | **0.716** | 20,224 |
| N19 | 1.75 | 120k | **0.174** | **0.091** | **0.511** | 19,885 |
| N20 | 2.0 | 120k | 0.128 | 0.081 | 0.707 | 17,994 |

**Curve is jagged, not monotonic/U-shaped:** 1.75 (N19) is the worst of the batch, worse even than 1.0 (N11), sitting between two better results. Caveat: single-step summary snapshots can be unlucky (individual steps can transiently blow up 100-1000x depending on the sampled frame) — N19 needs a visual check on `SWEEPN19_120k.ply` before concluding it's genuinely worst.

**For reference, the 60k results being compared against:**

| Arm | Weight | Duration | loss | L1 | SSIM | Gaussians |
|---|---|---|---|---|---|---|
| N5 | 1.0 | 60k | 0.088 | 0.065 | **0.834** | 47,705 |
| N12 | 1.5 | 60k | 0.144 | 0.088 | 0.647 | 42,109 |
| N13 | 2.0 | 60k | 0.076 | 0.047 | **0.819** | 40,952 |

**Verdict: extending to 120k does NOT reliably help or hurt — config-dependent.** Weight=1.5 improved with duration (0.647→0.716); weight=2.0 got worse (0.819→0.707); weight=1.0 got much worse (0.834→0.552, N11). No consistent direction, and **nothing in the 120k batch beats either 60k reference** — best results of the whole session remain 60k runs. Third and broadest confirmation raw duration isn't a dependable lever. Gaussian counts also shrink further by 120k across the board (all four sit 18k-23k, roughly half their 60k counterparts) — continued erosion under the 100%-densify-window pruning, not counteracted by higher anchor weight. `.ply`s: `SWEEPN17_{30,60,90,120}k.ply`, `SWEEPN18_{...}k.ply`, `SWEEPN20_{...}k.ply` — N19 pending.

## Metric-vs-visual-verdict correspondence, N1-N20 (2026-09-30)

**Tracked well:**
- N11 "really blurry sucks" (120k) ↔ ssim 0.552 vs. N5's 0.834, worst number in the set.
- N4 "no details, so sparse" ↔ max_scale 0.047-0.063 vs. ~0.29 for every other arm.
- N16 "features pruned, skewed/spiky" ↔ 15-20% fewer Gaussians than N13 at matched checkpoints.
- N15 "really bad, nuke it" ↔ real L1 spikes in wandb history (0.060→0.341 in one step).

**Did NOT track:**
- N1-N4: user ranked N2 best, N1 "decent" — but N1 has the *highest* SSIM of the four (0.706 vs. N2's 0.674).
- N12 vs. N13: user called both "not bad, smooth and clear" — but N12's SSIM (0.647) is far below N13's (0.819).
- N5 vs. N13 ("at par"): SSIM matched (0.834 vs. 0.819, ~1.8% apart) but L1/loss diverged sharply (N13 13-28% lower) with no matching visual gap.

**Conclusion:** SSIM is the best of the three metrics tracked but still an unreliable predictor alone — nailed N11's collapse, missed N1-vs-N2 and N12-vs-N13. Structural/population signals (Gaussian count, max_scale) are more consistent for explaining specific symptoms than aggregate loss/SSIM. Raw L1/loss is the least trustworthy — same lesson as the PSNR-misleading finding (`feedback_psnr_misleading_trust_visual_inspection.md`), generalized further. No single number substitutes for the render.

## Open / not yet tried
- **Frame-level exclusion** for the near-camera floater artifact (one video frame, 12,364 Gaussians collapsed within 0.3 units of that camera — motion blur/lens occlusion suspected). `--min_conf_frac` mean-confidence filtering didn't catch it (that frame's average confidence wasn't low enough). Needs a stricter/different per-frame criterion.
- **Densification aggressiveness** (`densify_grad_threshold=0.00008` vs. upstream `0.0002`, `opacity_cull=0.002` vs. upstream `0.005`) — ruled out as *dominant* cause (Run 3) but never re-tested against the current healthy baseline.
- Re-running COLMAP refinement (Run 8) combined with pose correction, now that pose correction alone is known to help structurally — untested combination.
