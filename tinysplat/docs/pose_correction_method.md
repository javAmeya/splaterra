# Differentiable Per-Camera Pose Correction

Implementation: `tinysplat/tinysplat/pose_correction.py` (`PoseCorrection`), integrated
into `tinysplat/train.py`. Empirically validated by direct visual inspection
(`docs/experiment_log.md`, Runs 9–10): near-elimination of parallax-duplicate
artifacts, at a PSNR cost relative to the uncorrected baseline. See
`docs/scaling_notes.md` for the broader investigation this sits within.

## 1. Formulation

Let $I_k$ denote the $k$-th of $N$ training images, and let
$\pi_k = (R_k, t_k) \in SE(3)$ denote its camera pose, where
$R_k \in SO(3)$ is a rotation matrix and $t_k \in \mathbb{R}^3$ a translation
vector, together giving the rigid transform from world to camera
coordinates. LoGeR produces $\pi_k$ for each image independently, inferring
it from a local temporal window of nearby frames with no mechanism enforcing
consistency across windows. For two frames $i, j$ that both observe some
shared physical point $X$, nothing constrains $\pi_i$ and $\pi_j$ to agree
on $X$'s position — each was inferred from a disjoint temporal context.

Vanilla 3DGS represents the scene as $G$ Gaussian primitives,
$\Theta = \{(\mu_g, \Sigma_g, \alpha_g, c_g)\}_{g=1}^{G}$, where for Gaussian
$g$: $\mu_g \in \mathbb{R}^3$ is its mean position, $\Sigma_g \in
\mathbb{R}^{3\times3}$ its covariance (size/orientation), $\alpha_g \in
[0,1]$ its opacity, and $c_g$ its color. Given a differentiable rasterizer
$\mathrm{Render}(\Theta, \pi)$ that renders $\Theta$ as seen from pose $\pi$,
and a per-image photometric loss $\ell(\cdot,\cdot)$ (L1 + D-SSIM in this
codebase — see `train.py`), training minimizes

$$\mathcal{L}(\Theta) = \mathbb{E}_{k \sim \mathcal{U}(1,N)}\left[\, \ell\big(\mathrm{Render}(\Theta, \pi_k),\, I_k\big) \,\right]$$

where $\mathcal{U}(1,N)$ is the uniform distribution over training indices
$\{1,\dots,N\}$, and every $\pi_k$ is held fixed throughout training.
Whenever $\pi_i$ and $\pi_j$ disagree about the position of a point $X$ they
both observe, no single $\Theta$ minimizes $\ell$ for both $i$ and $j$
simultaneously with respect to $X$ — the optimum instead is a compromise
geometry that partially satisfies each, observed as ghosted or duplicated
structure under novel-view parallax.

## 2. Parameterization

We assign every training camera $k$ a learnable correction
$\delta_k = (\omega_k, \tau_k) \in \mathbb{R}^6$, where $\omega_k \in
\mathbb{R}^3$ is an axis-angle rotation delta and $\tau_k \in \mathbb{R}^3$ a
translation delta. $\omega_k$ is converted to an actual rotation matrix via
the $SO(3)$ exponential map (Rodrigues' formula):

$$\mathrm{Exp}(\omega) = I_3 + \sin\theta\,[\hat\omega]_\times + (1-\cos\theta)\,[\hat\omega]_\times^2$$

where $I_3$ is the $3\times3$ identity matrix, $\theta = \lVert\omega\rVert$
is the rotation angle, $\hat\omega = \omega/\theta$ the unit rotation axis,
and $[\hat\omega]_\times$ the skew-symmetric matrix of $\hat\omega$ defined
so that $[\hat\omega]_\times v = \hat\omega \times v$ for any $v \in
\mathbb{R}^3$. The corrected pose $\pi_k' = (R_k', t_k')$ is then

$$R_k' = \mathrm{Exp}(\omega_k)\,R_k, \qquad t_k' = \mathrm{Exp}(\omega_k)\,t_k + \tau_k$$

At initialization $\delta_k = 0$, so $\mathrm{Exp}(0) = I_3$ and
$\pi_k' \equiv \pi_k$: training begins exactly at LoGeR's original poses and
departs from them only where the photometric gradient supports it.
`render()` accepts an optional `view_matrix` argument substituting $\pi_k'$
for $\pi_k$, so $\nabla_{\delta_k}\mathcal{L}$ (the gradient of the training
objective with respect to $\delta_k$) is available through the same
backward pass that already computes $\nabla_\Theta\mathcal{L}$ — no separate
optimization stage.

## 3. Optimization: stochastic, single-view, non-joint

$\mathcal{L}$ is never evaluated over more than one training image at once.
At training iteration $s$ (of $T$ total iterations), a single index $k_s$ is
sampled, $k_s \sim \mathcal{U}(1,N)$, only $I_{k_s}$ is rendered, and the
update applied is

$$\Theta \leftarrow \Theta - \eta_\Theta(s)\,\nabla_\Theta\, \ell\big(\mathrm{Render}(\Theta, \pi_{k_s}'),\, I_{k_s}\big)$$

$$\delta_{k_s} \leftarrow \delta_{k_s} - \eta_\delta(s)\,\nabla_{\delta_{k_s}}\,\ell(\cdot)$$

where $\eta_\Theta(s)$ and $\eta_\delta(s)$ are the (possibly iteration-
dependent) learning rates for $\Theta$ and for the pose corrections. For
every $j \neq k_s$, $\delta_j$ receives **exactly zero gradient** at
iteration $s$ — pose correction is updated one camera at a time, never
jointly across the cameras that happen to co-observe a given region.
Consistency between frames $i$ and $j$ is therefore never imposed at any
single iteration; it emerges only because $\Theta$ is a shared, persistent
variable across iterations. A Gaussian $g$ visible from both $\pi_i'$ and
$\pi_j'$ is pulled by frame $i$'s loss whenever $k_s = i$, and independently
pulled by frame $j$'s loss whenever $k_s = j$ — disjoint, generally
non-adjacent iterations. Over $T \gg N$ iterations, this alternating,
asynchronous updating of the shared variable $g$ (and correspondingly of
$\delta_i, \delta_j$) is what drives two otherwise independently-estimated
frames toward a mutually consistent geometry — an implicit,
temporally-distributed consensus, not a joint multi-view update.

## 4. Correspondence is geometric, not established

No iteration ever identifies a specific pixel $p$ in image $I_i$ and a
specific pixel $q$ in image $I_j$ as both observing the same physical point
$X$. Frames $i$ and $j$ are coupled *only if* some Gaussian $g \in \Theta$
currently lies inside both cameras' view frustums — write $F(\pi)$ for the
region of 3D space visible from pose $\pi$, so the condition is
$g \in F(\pi_i') \cap F(\pi_j')$. This holds only because LoGeR's initial
point cloud already placed $g$ near $X$ in both frames' own unprojections,
and $\pi_i, \pi_j$ already agree closely enough for the two frustums to
actually overlap there. This is a **local basin-of-attraction condition**:
the method can refine disagreement between frames whose current geometry
already brings them into approximate contact, but it cannot discover or
repair a correspondence that the current $(\Theta, \pi)$ estimate does not
already approximately represent.

This is the operative distinction from feature-based bundle adjustment
(`tinysplat/colmap_refine.py`): there, a correspondence $p \leftrightarrow
q$ — a verified claim that pixel $p$ in one image and pixel $q$ in another
observe the same physical point — is established by SIFT matching directly
on image content, independent of the current pose/geometry estimate, and
remains a valid correction signal even where frustum overlap under the
current estimate is poor or absent entirely.

## 5. Train/test asymmetry

Write $\mathcal{K}_{\text{train}}$ and $\mathcal{K}_{\text{test}}$ for the
training-camera and test-camera index sets, with $\mathcal{K}_{\text{train}}
\cup \mathcal{K}_{\text{test}} = \{1,\dots,N\}$. $\delta_k$ is defined only
for $k \in \mathcal{K}_{\text{train}}$; for $k \in \mathcal{K}_{\text{test}}$,
rendering always uses the original, unmodified $\pi_k$. A correction fit to
one specific training image has no analogue at a held-out viewpoint —
admitting it at eval time would condition on information no genuinely novel
view (e.g. in a deployed, walkable scene) could ever have available. Same
failure mode as the `use_trained_exp` bug (`experiment_log.md`, Run 5): never
optimize against a condition eval can't also have.

## 6. Configuration

Two runs, differing only in the learning-rate schedule $\eta_\delta(s)$
(both use a constant $\eta_\Theta(s)$ inherited from stock 3DGS):

| run | $\eta_\delta(1)$ | $\eta_\delta(T)$ | schedule | active from |
|---|---|---|---|---|
| v1 | $5\times10^{-4}$ | constant (no decay) | none | $s=1$ |
| v2 | $2\times10^{-3}$ | $2\times10^{-4}$ | exponential decay | $s=1$ |

Neither run delays when $\delta_k$ is first admitted into the optimization
until $\Theta$ has partially converged. A coarse-to-fine or warmup schedule
(as used in BARF, NeRF$--$) is not implemented here; a $4\times$ larger
$\eta_\delta(1)$ acting on gradients computed against an early, unconverged
$\Theta$ is a plausible mechanism for v2's non-monotonic PSNR trajectory
(`experiment_log.md`, Run 10).

## 7. Open directions

- **Warmup**: delay $\delta_k$'s admission until $\Theta$ reaches a coarse
  fit under the unmodified poses.
- **Temporal prior**: $\delta_k$ and $\delta_{k+1}$ are currently fully
  independent despite consecutive frames sharing one continuous underlying
  camera trajectory; a penalty on $\lVert\delta_k - \delta_{k+1}\rVert$,
  the magnitude of the difference between adjacent frames' corrections, is
  untried.
- **Composition with bundle adjustment**: running `colmap_refine.py` first,
  then continuing to refine jointly with $\Theta$ via pose correction during
  training, is untested. Explicit, correspondence-based bundle adjustment
  could supply the frustum-overlap precondition (§4) more reliably than
  LoGeR's raw poses alone.
