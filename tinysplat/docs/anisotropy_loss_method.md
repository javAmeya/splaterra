# Differentiable Anisotropy Penalty

Post-hoc pruning of needle-shaped Gaussians (Runs 13-16) always collapsed
the population, since pruning after densification ends has no
replenishment. This penalty was introduced to discourage needle shapes
from forming during training via gradient instead, avoiding that failure
mode entirely. Disabled (weight 0) in N5.

Implementation: `tinysplat/tinysplat/gaussian_model.py` (`anisotropy_loss`).

## 1. Formulation

For Gaussian $g$ with per-axis scale $s_g \in \mathbb{R}^3_{>0}$
(`get_scaling`, i.e. $\exp(\text{scales})$ — always positive by
construction), sort descending: $s_g^{(1)} \geq s_g^{(2)} \geq s_g^{(3)}$.
Define the **largest-to-second-largest** ratio:

$$r_g = \frac{s_g^{(1)}}{\max(s_g^{(2)}, \epsilon)}, \quad \epsilon = 10^{-8}$$

Using the ratio of the *top two* axes (not, say, largest-to-smallest) is
what distinguishes a genuine needle (one dominant axis, $r_g$ large) from a
legitimate flat/disk-shaped Gaussian (two comparable large axes, so
$r_g \approx 1$ regardless of how small the third axis is). A disk with
axes $(1.0, 0.9, 0.01)$ has $r_g \approx 1.11$ — untouched; a needle with
axes $(1.0, 0.05, 0.05)$ has $r_g = 20$ — heavily penalized.

The penalty is a **hinge/margin loss** — zero below a threshold, linear
above it:

$$\ell(g) = \max(0,\; r_g - \tau), \qquad \mathcal{L}_{\text{aniso}} = \frac{1}{G}\sum_{g} \ell(g)$$

with $\tau = 5.0$ (`anisotropy_ratio_threshold`) by default. Because it's a
hinge, Gaussians already below the threshold get **exactly zero gradient**
from this term — on this scene, the measured ratio distribution has median
$\approx 1.5$ and p90 $\approx 3.8$ (Runs 24-26 measurement), so the bulk of
the population sits well clear of $\tau=5.0$ and is completely unaffected;
only the tail of already-elongating Gaussians is pushed back.

Added to the total loss with weight $\lambda_{\text{aniso}}$
(`anisotropy_loss_weight`, $0.0$ in N5, i.e. off):

$$\mathcal{L}_{\text{total}} \mathrel{+}= \lambda_{\text{aniso}}\,\mathcal{L}_{\text{aniso}}$$

Like anchor loss, this is computed and its gradient applied on **every**
inner view (not gated on which camera is sampled) — cheap to evaluate (a
sort plus a few elementwise ops over the existing scale tensor, no extra
forward/render pass).

## 2. Empirical results — real tradeoff, not a free win

Turning the penalty on lowers `anisotropy_loss` (fewer needle shapes) but
worsens l1/ssim; turning it off does the opposite — direct evidence the
penalty trades real sharpness for shape control, not a free improvement.
N5's full-duration densify window already achieves a low `max_scale` on
its own, making the penalty's spike-control job largely redundant, so its
sharpness cost isn't worth paying and it stays off.

**Net conclusion**: this mechanism works as designed (it measurably
suppresses needle formation) but at a real sharpness cost that wasn't worth
paying once the full densify window was already controlling floaters well
on its own. Not "broken" — just not currently the right lever for this
specific recipe.

## 3. Open directions

- **Lower weight, not just on/off**: only $\{0, 0.005\}$ (and the earlier,
  cruder pruning-threshold approach) have been tried — an intermediate
  weight (e.g. 0.001-0.002) that suppresses only the most extreme needles
  while conceding less sharpness is untested.
- **Combine with N5's D+A recipe**: never tested together with depth
  supervision + anchor loss both on — possible the anchor tether alone
  already discourages needle formation as a side effect (an
  anchored-to-point-cloud Gaussian has less freedom to stretch arbitrarily
  far in one direction), which would make this penalty genuinely redundant
  in N5's specific recipe rather than just empirically not-worth-it.
