# EXP-E -- Hess-Flores et al.; Zhang et al. (arXiv:2008.01258), axis only.
#
# Fits once unweighted (identical to the baseline), then iteratively
# downweights views whose interpretation-plane normal disagrees most with the
# current direction estimate (residual = |normal . direction|, ~0 for a
# perfect fit since the direction should lie IN every interpretation plane)
# and re-triangulates, IterativeReweight.N_ITERATIONS times.
from __future__ import annotations

import numpy as np

from pipeline_common.triangulation import interpretation_plane_normal, ray_dir_for_point, widest_pair

from ._shared import weighted_triangulate_line

SYMMETRY_TYPES = ("axis_sym",)
NEEDS_MESH = False

N_ITERATIONS_DEFAULT = 3
DECAY_DEFAULT = 3.0   # weight = exp(-DECAY * residual); higher = harsher penalty on disagreement


def estimate_axis(
    points_by_image: dict, images_sent: list[dict], fov_deg: float, image_size: int,
    n_iterations: int = N_ITERATIONS_DEFAULT, decay: float = DECAY_DEFAULT,
) -> dict:
    """ Axis triangulation with iterative residual-based reweighting.

    Args:
        * points_by_image: Molmo2 points, one list per view index (string keys)
        * images_sent: camera pose entries (R/T), one per view, aligned by index
        * fov_deg: camera field of view in degrees
        * image_size: render size in pixels (square)
        * n_iterations: number of reweight-and-refit passes after the initial
          unweighted fit
        * decay: controls how harshly a high-residual view gets downweighted

    Returns:
        * dict: {"direction", "origin", "n_views_used"}

    """
    centers, normals = [], []
    for img_idx_str, pts in points_by_image.items():
        pair = widest_pair(pts)
        if pair is None:
            continue
        p_a, p_b = pair
        cam = images_sent[int(img_idx_str)]
        c, d_a = ray_dir_for_point(p_a["x"], p_a["y"], cam["R"], cam["T"], fov_deg, image_size)
        _, d_b = ray_dir_for_point(p_b["x"], p_b["y"], cam["R"], cam["T"], fov_deg, image_size)
        n = interpretation_plane_normal(d_a, d_b)
        if n is None:
            continue
        centers.append(c)
        normals.append(n)

    if len(normals) < 2:
        raise ValueError(f"need >=2 valid views, got {len(normals)}")

    normals_arr = np.asarray(normals, dtype=np.float64)
    weights = [1.0] * len(normals)
    point, direction = weighted_triangulate_line(centers, normals, weights)

    for _ in range(n_iterations):
        residuals = np.abs(normals_arr @ direction)
        weights = np.exp(-decay * residuals).tolist()
        point, direction = weighted_triangulate_line(centers, normals, weights)

    return {"direction": direction.tolist(), "origin": point.tolist(), "n_views_used": len(normals)}
