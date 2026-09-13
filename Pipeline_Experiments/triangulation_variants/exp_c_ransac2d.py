# EXP-C -- Recker et al. (WACV 2013).
#
# Replaces the point-pair selector (widest_pair / role_pair_selector) with a
# RANSAC 2D line fit + widest-inlier-pair pick (ransac_line_2d), and keeps the
# baseline's UNWEIGHTED triangulation (uniform_weight) -- isolates the effect
# of robust point selection from any confidence weighting. With today's
# 2-point prompts this is a deterministic no-op; it starts to matter once a
# view returns >=3 candidate points (the "6pts" prompt family), where it
# discards points that don't lie near the consensus line first.
from __future__ import annotations

from ._shared import generic_detect_planes, generic_estimate_axis, generic_estimate_plane, ransac_line_2d, uniform_weight

SYMMETRY_TYPES = ("axis_sym", "plane_sym")
NEEDS_MESH = False


def estimate_axis(points_by_image: dict, images_sent: list[dict], fov_deg: float, image_size: int) -> dict:
    """ Axis triangulation with RANSAC-selected point pairs.

    Args:
        * points_by_image: Molmo2 points, one list per view index (string keys)
        * images_sent: camera pose entries (R/T), one per view, aligned by index
        * fov_deg: camera field of view in degrees
        * image_size: render size in pixels (square)

    Returns:
        * dict: {"direction", "origin", "n_views_used"}

    """
    return generic_estimate_axis(points_by_image, images_sent, fov_deg, image_size, ransac_line_2d, uniform_weight)


def estimate_plane(
    points_by_image: dict, images_sent: list[dict], fov_deg: float, image_size: int, edge_on_thresh: float,
) -> dict:
    """ Single-plane estimation with RANSAC-selected point pairs.

    Ignores obj_id roles entirely (unlike the baseline's fixed 1/2 lookup):
    any view with >=2 candidate points qualifies, and the widest RANSAC
    inlier pair stands in for the bilateral role pair.

    Args:
        * points_by_image: Molmo2 points, one list per view index (string keys)
        * images_sent: camera pose entries (R/T), one per view, aligned by index
        * fov_deg: camera field of view in degrees
        * image_size: render size in pixels (square)
        * edge_on_thresh: |cos(angle)| below this counts a view as "edge-on"

    Returns:
        * dict: {"normal", "origin", "n_views_used", "n_candidates", "good_views"}

    """
    return generic_estimate_plane(points_by_image, images_sent, fov_deg, image_size, edge_on_thresh,
                                   ransac_line_2d, uniform_weight)


def detect_planes(
    points_by_image: dict, images_sent: list[dict], fov_deg: float, image_size: int,
    edge_on_thresh: float, max_planes: int, dup_angle_thresh_deg: float, mesh_ctx: dict | None = None,
) -> list[dict]:
    """ Sequential multi-plane consolidation with RANSAC-selected point pairs.

    Args:
        * points_by_image: Molmo2 points, one list per view index (string keys)
        * images_sent: camera pose entries (R/T), one per view, aligned by index
        * fov_deg: camera field of view in degrees
        * image_size: render size in pixels (square)
        * edge_on_thresh: |cos(angle)| below this counts a view as "edge-on"
        * max_planes: stop once this many planes have been accepted
        * dup_angle_thresh_deg: candidate planes closer than this are duplicates
        * mesh_ctx: unused (this variant needs no mesh access)

    Returns:
        * list[dict]: accepted planes, in acceptance order

    """
    return generic_detect_planes(points_by_image, images_sent, fov_deg, image_size, edge_on_thresh,
                                  max_planes, dup_angle_thresh_deg, ransac_line_2d, uniform_weight)
