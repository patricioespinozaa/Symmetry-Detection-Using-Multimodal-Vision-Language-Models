# EXP-B -- Wu et al. (PMC, 2021), axis only.
#
# "Point-then-direction": the baseline's triangulate_line solves for a point
# on the line and the line's direction from the SAME weighted lstsq system.
# This variant decouples them -- the anchor is the least-squares point
# closest to every individual ray used by the baseline's pairs (weighted by
# each view's pixel separation), computed independently of the direction fit
# (a weighted-SVD null direction, same formula as EXP-A). No published
# reference code exists for this on 3D-point/VLM pointing (the paper's domain
# is hand-pose triangulation) -- this is our own concrete instantiation of
# the "separate anchor from direction" idea, not a verbatim port.
from __future__ import annotations

from pipeline_common.triangulation import interpretation_plane_normal, ray_dir_for_point, widest_pair

from ._shared import closest_point_to_rays, pixel_length, weighted_null_direction

SYMMETRY_TYPES = ("axis_sym",)
NEEDS_MESH = False


def estimate_axis(points_by_image: dict, images_sent: list[dict], fov_deg: float, image_size: int) -> dict:
    """ Axis triangulation with the anchor point and direction fit independently.

    Args:
        * points_by_image: Molmo2 points, one list per view index (string keys)
        * images_sent: camera pose entries (R/T), one per view, aligned by index
        * fov_deg: camera field of view in degrees
        * image_size: render size in pixels (square)

    Returns:
        * dict: {"direction", "origin", "n_views_used"}

    """
    ray_origins, ray_dirs, ray_weights = [], [], []
    plane_centers, plane_normals, plane_weights = [], [], []

    for img_idx_str, pts in points_by_image.items():
        pair = widest_pair(pts)
        if pair is None:
            continue
        p_a, p_b = pair
        cam = images_sent[int(img_idx_str)]
        center, d_a = ray_dir_for_point(p_a["x"], p_a["y"], cam["R"], cam["T"], fov_deg, image_size)
        _, d_b = ray_dir_for_point(p_b["x"], p_b["y"], cam["R"], cam["T"], fov_deg, image_size)
        weight = pixel_length(p_a, p_b)

        ray_origins += [center, center]
        ray_dirs += [d_a, d_b]
        ray_weights += [weight, weight]

        normal = interpretation_plane_normal(d_a, d_b)
        if normal is None:
            continue
        plane_centers.append(center)
        plane_normals.append(normal)
        plane_weights.append(weight)

    if len(plane_normals) < 2:
        raise ValueError(f"need >=2 valid views, got {len(plane_normals)}")

    anchor = closest_point_to_rays(ray_origins, ray_dirs, ray_weights)
    direction = weighted_null_direction(plane_normals, plane_weights)
    return {"direction": direction.tolist(), "origin": anchor.tolist(), "n_views_used": len(plane_normals)}
