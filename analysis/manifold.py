"""Manifold geometry metrics: few-shot SNR and replica mean-field capacity.

Ported near-verbatim from visreps/experiments/manifold_analysis/manifold_snr.py
and manifold_capacity.py (Yash Mehta's implementation of Sorscher et al. 2022
and Chung et al. 2018 respectively). Both are model-agnostic numpy functions
operating on pre-extracted activation "manifolds" -- no visreps-specific
dependencies, so nothing else needed changing to port them.

`manifold_capacity` requires the authors' reference package (not on PyPI):
    pip install git+https://github.com/schung039/neural_manifolds_replicaMFT.git
(already in this repo's requirements.txt).
"""
from __future__ import annotations

import inspect

import numpy as np
from numpy.typing import ArrayLike


def manifold_snr(manifolds: ArrayLike, n_shots: int = 5) -> dict[str, object]:
    """Compute directed pairwise few-shot SNRs (Sorscher et al. 2022) and their mean.

    Args:
        manifolds: Array shaped (n_categories, n_features, n_images).
        n_shots: Number of examples per category in the hypothetical prototype
            classifier. The defining DNN analyses use five.

    Returns:
        ``mean`` is the mean over all ordered category pairs. ``pairwise`` is
        the directed (n_categories, n_categories) matrix with a NaN diagonal.
    """
    x = np.asarray(manifolds, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("manifolds must have shape (categories, features, images)")
    if x.shape[0] < 2 or x.shape[2] < 2:
        raise ValueError("at least two categories and two images are required")
    if not isinstance(n_shots, (int, np.integer)) or n_shots < 1:
        raise ValueError("n_shots must be a positive integer")
    if not np.isfinite(x).all():
        raise ValueError("manifolds contains NaN or infinite values")

    n_categories, _, n_images = x.shape
    centers = x.mean(axis=2)

    # The reference code applies SVD to (images, features). Vh therefore holds
    # feature-space principal axes and s holds the within-manifold radii.
    radii = []
    axes = []
    for category in x:
        _, singular_values, vh = np.linalg.svd(
            (category - category.mean(axis=1, keepdims=True)).T,
            full_matrices=False,
        )
        radii.append(singular_values)
        axes.append(vh)
    radii = np.stack(radii)
    axes = np.stack(axes)

    radius_sq_sum = np.sum(radii**2, axis=1)
    if np.any(radius_sq_sum <= 0):
        raise ValueError("every category must have nonzero within-category variance")
    dimensions = radius_sq_sum**2 / np.sum(radii**4, axis=1)

    center_delta = centers[:, None, :] - centers[None, :, :]
    center_distance = np.linalg.norm(center_delta, axis=2)
    normalized_distance = center_distance / np.sqrt(
        radius_sq_sum[:, None] / n_images
    )

    center_self = np.full((n_categories, n_categories), np.nan)
    center_other = np.full_like(center_self, np.nan)
    subspace_overlap = np.full_like(center_self, np.nan)

    for a in range(n_categories):
        for b in range(n_categories):
            if a == b:
                continue
            if center_distance[a, b] == 0:
                raise ValueError(f"categories {a} and {b} have identical centers")

            direction = center_delta[a, b] / center_distance[a, b]
            center_self[a, b] = np.sum(
                (axes[a] @ direction) ** 2 * radii[a] ** 2
            ) / radius_sq_sum[a]
            center_other[a, b] = np.sum(
                (axes[b] @ direction) ** 2 * radii[b] ** 2
            ) / radius_sq_sum[a]

            axis_cosines = axes[a] @ axes[b].T
            subspace_overlap[a, b] = np.sum(
                axis_cosines**2
                * radii[a, :, None] ** 2
                * radii[b, None, :] ** 2
            ) / radius_sq_sum[a] ** 2

    signal_noise_overlap = normalized_distance**2 * (
        center_self + center_other / n_shots
    )
    bias = radius_sq_sum[None, :] / radius_sq_sum[:, None] - 1
    pairwise = 0.5 * (normalized_distance**2 + bias / n_shots) / np.sqrt(
        1 / dimensions[:, None] / n_shots
        + signal_noise_overlap
        + subspace_overlap / n_shots
    )
    np.fill_diagonal(pairwise, np.nan)

    return {
        "mean": float(np.nanmean(pairwise)),
        "pairwise": pairwise,
        "dimension": dimensions,
        "radius": np.sqrt(radius_sq_sum / n_images),
    }


def manifold_capacity(
    manifolds: ArrayLike,
    *,
    margin: float = 0.0,
    n_probes: int = 200,
    seed: int = 0,
) -> dict[str, object]:
    """Compute correlated-manifold classification capacity (Chung et al. 2018).

    Calls the authors' reference implementation (`mftma`) rather than a
    numerically different reimplementation.

    Args:
        manifolds: Array shaped (n_categories, n_features, n_images). Requires
            n_features > n_images (the augmented per-manifold span must stay
            below full rank).
        margin: Classification margin. The paper uses zero.
        n_probes: Gaussian probes per manifold. The reference default is 200.
        seed: Seed for factor analysis and Gaussian probes.

    Returns:
        ``mean`` is the required harmonic mean of per-manifold capacities.
    """
    x = np.asarray(manifolds, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("manifolds must have shape (categories, features, images)")
    if x.shape[0] < 2 or x.shape[2] < 2:
        raise ValueError("at least two categories and two images are required")
    if x.shape[1] <= x.shape[2]:
        raise ValueError("capacity requires n_features > n_images")
    if not np.isfinite(x).all():
        raise ValueError("manifolds contains NaN or infinite values")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    if not isinstance(n_probes, (int, np.integer)) or n_probes < 1:
        raise ValueError("n_probes must be a positive integer")

    # The reference package targets older Python/SciPy APIs. These aliases only
    # restore removed names; they do not alter the numerical implementation.
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    import scipy.misc
    import scipy.special

    if not hasattr(scipy.misc, "comb"):
        scipy.misc.comb = scipy.special.comb  # type: ignore[attr-defined]

    try:
        from mftma.manifold_analysis_correlation import manifold_analysis_corr
    except ImportError as error:
        raise ImportError(
            "manifold capacity requires the authors' neural_manifolds_replicaMFT "
            "package: pip install git+https://github.com/schung039/"
            "neural_manifolds_replicaMFT.git"
        ) from error

    random_state = np.random.get_state()
    try:
        np.random.seed(seed)
        capacity, radius, dimension, center_correlation, center_rank = (
            manifold_analysis_corr(
                [category for category in x],
                kappa=float(margin),
                n_t=int(n_probes),
            )
        )
    finally:
        np.random.set_state(random_state)
    capacity = np.asarray(capacity, dtype=np.float64)
    if np.any(~np.isfinite(capacity)) or np.any(capacity <= 0):
        raise RuntimeError("reference implementation returned invalid capacities")

    return {
        "mean": float(1 / np.mean(1 / capacity)),
        "per_manifold": capacity,
        "radius": np.asarray(radius),
        "dimension": np.asarray(dimension),
        "center_correlation": float(center_correlation),
        "center_rank": int(center_rank),
    }
