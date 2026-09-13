# EXP-A -- Bartoli & Sturm (CVIU 2005, DOI:10.1016/j.cviu.2005.06.001).
#
# Weights every interpretation plane / pair-line by the 2D pixel separation
# of the two points that built it, instead of the baseline's unweighted SVD
# (every view counts equally regardless of how reliable its line is). Applies
# to both axis and plane; see Experiments/sandbox_pipeline_server_extended.ipynb.
from __future__ import annotations

from pipeline_common.triangulation import widest_pair

from ._shared import generic_estimate_axis, generic_estimate_plane, generic_detect_planes, pixel_length, role_pair_selector

SYMMETRY_TYPES = ("axis_sym", "plane_sym")
NEEDS_MESH = False


def estimate_axis(points_by_image: dict, images_sent: list[dict], fov_deg: float, image_size: int) -> dict:
    """ Axis triangulation weighted by per-view pixel separation.

    Args:
        * points_by_image: Molmo2 points, one list per view index (string keys)
        * images_sent: camera pose entries (R/T), one per view, aligned by index
        * fov_deg: camera field of view in degrees
        * image_size: render size in pixels (square)

    Returns:
        * dict: {"direction", "origin", "n_views_used"}

    """
    return generic_estimate_axis(points_by_image, images_sent, fov_deg, image_size, widest_pair, pixel_length)


def estimate_plane(
    points_by_image: dict, images_sent: list[dict], fov_deg: float, image_size: int, edge_on_thresh: float,
) -> dict:
    """ Single-plane estimation weighted by per-pair-line pixel separation.

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
                                   role_pair_selector, pixel_length)


def detect_planes(
    points_by_image: dict, images_sent: list[dict], fov_deg: float, image_size: int,
    edge_on_thresh: float, max_planes: int, dup_angle_thresh_deg: float, mesh_ctx: dict | None = None,
) -> list[dict]:
    """ Sequential multi-plane consolidation weighted by per-pair-line pixel separation.

    Args:
        * points_by_image: Molmo2 points, one list per view index (string keys)
        * images_sent: camera pose entries (R/T), one per view, aligned by index
        * fov_deg: camera field of view in degrees
        * image_size: render size in pixels (square)
        * edge_on_thresh: |cos(angle)| below this counts a view as "edge-on"
        * max_planes: stop once this many planes have been accepted
        * dup_angle_thresh_deg: candidate planes closer than this are duplicates
        * mesh_ctx: unused (this variant needs no mesh access), kept for
          interface parity with exp_f_sde_gate.detect_planes

    Returns:
        * list[dict]: accepted planes, in acceptance order

    """
    return generic_detect_planes(points_by_image, images_sent, fov_deg, image_size, edge_on_thresh,
                                  max_planes, dup_angle_thresh_deg, role_pair_selector, pixel_length)
