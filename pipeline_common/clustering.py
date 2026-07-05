"""
clustering.py
-------------
Point-cloud consolidation strategies applied before RANSAC+SVD fitting
(Mapping/estimate_symmetry.py), also used by the debug viewer
(Mapping/visualize_rays.py) to inspect the same clusters interactively.
"""
from __future__ import annotations

import numpy as np


def cluster_points(
    points: np.ndarray,
    bbox_diag: float,
    threshold_fraction: float = 0.05,
) -> np.ndarray:
    """
    Greedy centroid-based spatial clustering.

    Points within threshold_fraction * bbox_diag of an existing centroid are
    merged into it (centroid updated incrementally). Reduces spatial
    redundancy when multiple views observe the same surface region.

    Every input point is always assigned to some cluster -- there is no
    concept of an outlier here (contrast with cluster_hdbscan, which can
    discard noise points).

    Returns an array of cluster centroids (one per group), preserving order
    of first encounter. If threshold is too small or fewer than 2 points
    exist, returns the original array unchanged.
    """
    threshold = threshold_fraction * bbox_diag
    if threshold < 1e-8 or len(points) < 2:
        return points

    centroids: list[np.ndarray] = []
    counts: list[int] = []

    for pt in points:
        placed = False
        for k, c in enumerate(centroids):
            if np.linalg.norm(pt - c) < threshold:
                counts[k] += 1
                centroids[k] = c + (pt - c) / counts[k]   # incremental mean
                placed = True
                break
        if not placed:
            centroids.append(pt.copy())
            counts.append(1)

    return np.array(centroids, dtype=np.float64)


def cluster_hdbscan(
    points: np.ndarray,
    min_samples: int = 3,
    min_cluster_size: int = 2,
) -> np.ndarray:
    """
    HDBSCAN-based density clustering.

    Unlike cluster_points, HDBSCAN explicitly labels sparse/isolated points as
    noise (label -1) and drops them before returning centroids -- useful for
    discarding points caused by Molmo hallucinations that happen to hit the
    mesh but land far from the rest of the cloud.

    min_cluster_size is fixed (not exposed as a user-facing sweep parameter):
    point clouds here are small (tens of points), and scikit-learn's own
    default of 5 would over-aggressively call everything noise at low
    n_views. min_samples is the parameter the methodology sweeps over
    (2, 3, 5).

    Falls back to returning the input points unchanged if there are fewer
    than 2 points, if min_samples/min_cluster_size can't be satisfied by the
    number of points (sklearn raises ValueError in that case rather than
    returning all-noise -- this happens routinely at small n_views groups),
    or if every point is labeled noise (degenerate input) -- mirrors
    cluster_points' own fallback behavior rather than crashing or silently
    producing an empty cloud that would cause the n_views group to be
    skipped downstream.
    """
    if len(points) < 2 or min_samples >= len(points) or min_cluster_size > len(points):
        return points

    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as e:
        raise ImportError(
            "cluster_hdbscan requires scikit-learn (sklearn.cluster.HDBSCAN, "
            "available since scikit-learn>=1.3). Install it with "
            "`pip install scikit-learn` (see environment.yml)."
        ) from e

    labels = HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
    ).fit_predict(points)

    unique_labels = sorted(l for l in set(labels) if l != -1)
    if not unique_labels:
        return points

    return np.array(
        [points[labels == l].mean(axis=0) for l in unique_labels],
        dtype=np.float64,
    )
