"""Fixed-dictionary compression utilities used by the v2 diagnostics.

The routines in this module implement the random construction appearing in
Lemma 8 of the manuscript.  They do not move first-layer atoms and preserve
the output-layer L1 mass exactly (up to floating-point roundoff).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import ModelState


@dataclass(frozen=True)
class SamplingBatch:
    coefficients: np.ndarray
    probabilities: np.ndarray
    atomic_mass: float


def sample_on_fixed_dictionary(
    theta: np.ndarray,
    q: int,
    trials: int,
    rng: np.random.Generator,
) -> SamplingBatch:
    """Draw ``trials`` penalty-preserving coefficient compressions.

    Each returned row has support at most ``q`` and the same L1 norm as the
    input.  Repeated atom indices are aggregated with a multinomial count.
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if not 1 <= q <= len(theta):
        raise ValueError("q must satisfy 1 <= q <= len(theta).")
    if trials < 1:
        raise ValueError("trials must be positive.")
    atomic_mass = float(np.linalg.norm(theta, ord=1))
    if atomic_mass == 0.0:
        return SamplingBatch(
            coefficients=np.zeros((trials, len(theta)), dtype=np.float64),
            probabilities=np.zeros_like(theta),
            atomic_mass=0.0,
        )
    probabilities = np.abs(theta) / atomic_mass
    counts = rng.multinomial(q, probabilities, size=trials)
    coefficients = counts * np.sign(theta)[None, :] * (atomic_mass / q)
    return SamplingBatch(coefficients, probabilities, atomic_mass)


def smallest_mass_pruning(state: ModelState, target_support: int) -> ModelState:
    """Keep the coefficients of largest absolute mass on the same atoms."""
    if not 0 <= target_support <= state.width:
        raise ValueError("target_support must lie between zero and the width.")
    if target_support == state.width:
        return state.copy()
    result = state.copy()
    active = np.flatnonzero(result.theta != 0.0)
    remove_count = max(0, len(active) - target_support)
    if remove_count:
        order = active[np.argsort(np.abs(result.theta[active]), kind="stable")]
        result.theta[order[:remove_count]] = 0.0
    return result


def balanced_weight_decay_lift(state: ModelState) -> ModelState:
    """Lift atomic coefficients to balanced raw two-layer parameters.

    If ``a_i`` and ``u_i`` are the atomic coefficient and atom, the raw
    parameters are ``sign(a_i)*sqrt(|a_i|)`` and
    ``sqrt(|a_i|)*u_i``.  The represented predictor and regularizer agree.
    """
    radii = np.sqrt(np.abs(state.theta))
    raw_theta = np.sign(state.theta) * radii
    raw_W = state.W * radii[:, None]
    return ModelState(raw_W, raw_theta)


def raw_weight_decay_penalty(state: ModelState, kappa: float) -> float:
    return float(
        0.5
        * kappa
        * (np.sum(state.theta**2) + np.sum(state.W**2))
    )
