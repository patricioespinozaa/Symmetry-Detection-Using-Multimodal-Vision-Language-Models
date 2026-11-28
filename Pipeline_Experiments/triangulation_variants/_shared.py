from __future__ import annotations

import numpy as np

# Weighted linear algebra used by EXP-E (exp_e_iterative_reweight.py). The other
# ablations (EXP-A..D, F) and their helpers were removed from the repo because the
# thesis only reports SVD vs. EXP-E; they remain available in the git history.


def weighted_null_direction(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """ Unit vector minimizing the weighted sum of squared dot-products with each row.

    Args:
        * rows: (k, 3) array, one row per observation (interpretation-plane
          normal, or candidate line direction)
        * weights: (k,) non-negative reliability weight per row

    Returns:
        * np.ndarray: (3,) unit vector -- the weighted counterpart of the
          unweighted SVD's smallest right-singular vector (see
          pipeline_common.triangulation.triangulate_line)

    """
    w = np.sqrt(np.clip(np.asarray(weights, dtype=np.float64), 1e-9, None))
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64) * w[:, None])
    direction = vt[-1]
    return direction / np.linalg.norm(direction)


def weighted_triangulate_line(
    camera_centers: list[np.ndarray], plane_normals: list[np.ndarray], weights: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """ Weighted counterpart of pipeline_common.triangulation.triangulate_line.

    Args:
        * camera_centers: one (3,) world-space camera center per interpretation plane
        * plane_normals: one (3,) unit normal per interpretation plane
        * weights: one non-negative reliability weight per interpretation plane

    Returns:
        * tuple[np.ndarray, np.ndarray]: (point_on_line, unit_direction)

    """
    n = np.asarray(plane_normals, dtype=np.float64)
    c = np.asarray(camera_centers, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    direction = weighted_null_direction(n, w)

    sqrt_w = np.sqrt(np.clip(w, 1e-9, None))
    a = n * sqrt_w[:, None]
    b = np.einsum("ij,ij->i", n, c) * sqrt_w
    point, *_ = np.linalg.lstsq(a, b, rcond=None)
    return point, direction
